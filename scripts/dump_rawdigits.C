// dump_rawdigits.C
//
// Gallery macro that dumps raw::RawDigit ADC waveforms from one or more
// LArSoft/art ROOT files into a *flat* TTree that uproot can read directly
// (no art dictionaries needed downstream).  This is the supported bridge
// between the art raw data and the Python VAE / GNN pipeline:
//
//     art RawDigit files  --(this macro)-->  flat ntuple  --(uproot)-->  Python
//
// Mirrors the access pattern of adchistos.C:
//     rd.Channel()        channel id
//     rd.Samples()        number of ADC ticks
//     rd.GetPedestal()    per-channel pedestal
//     rd.ADC(itick)       ADC value at a tick
//
// The output tree "rawdigits" has one entry per (event), with jagged,
// per-channel arrays.  Layout chosen so uproot returns awkward arrays:
//
//     run, subrun, event      : Int_t      (scalars)
//     nchan                   : Int_t      (channels in this event)
//     nticks                  : Int_t      (samples per channel; assumed uniform)
//     channel[nchan]          : Int_t
//     pedestal[nchan]         : Float_t
//     adc[nchan*nticks]       : Short_t    (row-major: channel-major, tick-minor)
//
// In Python (raw_digit_reader.py) adc is reshaped to (nchan, nticks).
//
// Usage (interactive ROOT with gallery + larsoft set up):
//     root -l -b -q 'dump_rawdigits.C("input_tpcdecode.root", "rawdigits.root", "daq", 650)'
//
// Or loop a file list from a shell:
//     for f in $(cat filelist.txt); do
//       root -l -b -q "dump_rawdigits.C(\"$f\", \"$(basename $f .root)_raw.root\", \"daq\", 0)"
//     done
//
// nevents = 0 means "all events in the file".

#include <iostream>
#include <string>
#include <vector>

#include "canvas/Utilities/InputTag.h"
#include "gallery/Event.h"

#include "TFile.h"
#include "TTree.h"

#include "lardataobj/RawData/RawDigit.h"

using namespace art;
using namespace std;

void dump_rawdigits(std::string const& filename = "input_tpcdecode.root",
                    std::string const& outputname = "rawdigits.root",
                    std::string const& inputtag = "daq",
                    size_t nevents = 0)
{
  InputTag rawdigit_tag{inputtag};
  vector<string> filenames(1, filename);

  // ---- Output flat ntuple ------------------------------------------------
  TFile outputfile(outputname.c_str(), "RECREATE");
  TTree* tree = new TTree("rawdigits", "Flat raw ADC dump for ML pipeline");

  Int_t run = 0, subrun = 0, event = 0;
  Int_t nchan = 0, nticks = 0;
  std::vector<Int_t> channel;
  std::vector<Float_t> pedestal;
  std::vector<Short_t> adc;  // flattened nchan*nticks, channel-major

  tree->Branch("run", &run, "run/I");
  tree->Branch("subrun", &subrun, "subrun/I");
  tree->Branch("event", &event, "event/I");
  tree->Branch("nchan", &nchan, "nchan/I");
  tree->Branch("nticks", &nticks, "nticks/I");
  tree->Branch("channel", &channel);
  tree->Branch("pedestal", &pedestal);
  tree->Branch("adc", &adc);

  size_t evcounter = 0;
  for (gallery::Event ev(filenames); !ev.atEnd(); ev.next()) {
    if (nevents != 0 && evcounter >= nevents) break;

    auto const& aux = ev.eventAuxiliary();
    run = static_cast<Int_t>(aux.run());
    subrun = static_cast<Int_t>(aux.subRun());
    event = static_cast<Int_t>(aux.event());

    auto const& rawdigits = *ev.getValidHandle<vector<raw::RawDigit>>(rawdigit_tag);

    channel.clear();
    pedestal.clear();
    adc.clear();
    nchan = 0;
    nticks = 0;

    if (!rawdigits.empty()) {
      // Assume a uniform sample count across channels (true for decoded TPC
      // data within a readout window).  Use the first channel to fix nticks.
      nticks = static_cast<Int_t>(rawdigits[0].Samples());
      nchan = static_cast<Int_t>(rawdigits.size());

      channel.reserve(nchan);
      pedestal.reserve(nchan);
      adc.reserve(static_cast<size_t>(nchan) * static_cast<size_t>(nticks));

      for (size_t ic = 0; ic < rawdigits.size(); ++ic) {
        auto const& rd = rawdigits[ic];
        channel.push_back(static_cast<Int_t>(rd.Channel()));
        pedestal.push_back(static_cast<Float_t>(rd.GetPedestal()));

        // RawDigit may be stored compressed; Uncompress() expands it.
        // ADC(itick) accessor handles uncompressed digits directly.
        const size_t nsamp = rd.Samples();
        for (size_t itick = 0; itick < nsamp; ++itick) {
          // Guard against ragged channels: pad/truncate to nticks.
          if (static_cast<Int_t>(itick) < nticks)
            adc.push_back(static_cast<Short_t>(rd.ADC(itick)));
        }
        // Pad short channels so the flat array stays rectangular.
        for (Int_t itick = static_cast<Int_t>(nsamp); itick < nticks; ++itick)
          adc.push_back(static_cast<Short_t>(rd.GetPedestal()));
      }
    }

    tree->Fill();
    ++evcounter;
  }

  outputfile.cd();
  tree->Write();
  outputfile.Close();

  std::cout << "Wrote " << evcounter << " event(s) to " << outputname
            << " (tree 'rawdigits')" << std::endl;
}

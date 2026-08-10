#include <regex>
#include <string>
#include <vector>
#include <iostream>
#include <set>
#include <utility>

using namespace std;


// ------------------------------------------------------------
// Good/bad classification Switch
// true  = good runs one color, bad runs another color
// false = each run gets its own color
// ------------------------------------------------------------

bool color_by_good_bad = true;


// ------------------------------------------------------------
// Channel selection settings
// Use [channel_min, channel_max), meaning:
// channel_min included, channel_max excluded
//
// This macro plots hits2.h.channel itself,
// but only for hits whose channel is inside the requested range.
// ------------------------------------------------------------

bool use_channel_cut = true;

vector<pair<int, int>> channel_ranges = {
    // {0, 11276}, // full range

    {3900, 5700},
    {9600, 11276}
};


// ------------------------------------------------------------
// List of run directories
// First one is the reference run
// ------------------------------------------------------------

vector<const char*> run_dirs = {
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_19305/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_19308/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_19315/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_829/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20769/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20782/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20768/reco/",

    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20614/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20615/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20620/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20621/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20173/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_830/reco/",
};


// ------------------------------------------------------------
// Good/bad classification Sets
// ------------------------------------------------------------

set<int> good_runs = {
    19305, 19308, 19315, 829, 20769, 20782, 20768
};

set<int> bad_runs = {
    20614, 20615, 20620, 20621, 20173, 830
};


// ------------------------------------------------------------
// Color palettes
// Good runs use cold colors.
// Bad runs use hot colors.
// ------------------------------------------------------------

vector<int> good_colors = {
    TColor::GetColor("#2483c8"), // blue
    TColor::GetColor("#063d6b"), // medium blue
    TColor::GetColor("#17becf"), // cyan
    TColor::GetColor("#00a087"), // teal
    TColor::GetColor("#2ca02c"), // green
    TColor::GetColor("#4daf4a"), // medium green
    TColor::GetColor("#66c2a5")  // pale teal
};

vector<int> bad_colors = {
    TColor::GetColor("#d62728"), // red
    TColor::GetColor("#e41a1c"), // bright red
    TColor::GetColor("#b2182b"), // dark red
    TColor::GetColor("#ff7f0e"), // orange
    TColor::GetColor("#a65628"), // brown-orange
    TColor::GetColor("#613807")  // dark brown-orange
};

int unknown_color = kBlack;


// ------------------------------------------------------------
// Extract run number from path
// ------------------------------------------------------------

int extract_run_number(const char* path) {
    string s(path);

    regex pattern("CI_build_lar_ci_([0-9]+)");
    smatch match;

    if (regex_search(s, match, pattern)) {
        return stoi(match[1]);
    }

    return -1;
}


// ------------------------------------------------------------
// Decide color from good/bad label
// ------------------------------------------------------------

int get_run_color(int run) {
    if (good_runs.count(run)) {
        int index = 0;

        for (const char* dir : run_dirs) {
            int r = extract_run_number(dir);

            if (!good_runs.count(r)) continue;

            if (r == run) {
                return good_colors[index % good_colors.size()];
            }

            index++;
        }
    }

    if (bad_runs.count(run)) {
        int index = 0;

        for (const char* dir : run_dirs) {
            int r = extract_run_number(dir);

            if (!bad_runs.count(r)) continue;

            if (r == run) {
                return bad_colors[index % bad_colors.size()];
            }

            index++;
        }
    }

    return unknown_color;
}


// ------------------------------------------------------------
// Decide label from good/bad label
// ------------------------------------------------------------

const char* get_run_status(int run) {
    if (good_runs.count(run)) return "good";
    if (bad_runs.count(run)) return "bad";
    return "unknown";
}


// ------------------------------------------------------------
// Add only readable ROOT files that contain the requested tree
// ------------------------------------------------------------

int add_good_files_to_chain(TChain* chain, const char* dir, const char* treePath) {
    TString command = Form("ls %s/DQMValidationTrees_*.root 2>/dev/null", dir);
    TString file_list = gSystem->GetFromPipe(command);

    TObjArray* lines = file_list.Tokenize("\n");

    int n_added = 0;

    for (int i = 0; i < lines->GetEntries(); i++) {
        TString file_path = lines->At(i)->GetName();

        if (file_path.Length() == 0) continue;

        // Optional: skip known hanging files before TFile::Open.
        // Add exact file-name patterns here if needed.
        /*
        if (file_path.Contains("DQMValidationTrees_011.root") ||
            file_path.Contains("DQMValidationTrees_11.root")) {
            cout << "Skipping known hanging file: " << file_path << endl;
            continue;
        }
        */

        TFile* f = TFile::Open(file_path);

        if (!f || f->IsZombie()) {
            cout << "Skipping bad/unreadable file: " << file_path << endl;
            if (f) f->Close();
            continue;
        }

        TTree* t = (TTree*)f->Get(treePath);

        if (!t) {
            cout << "Skipping file without tree " << treePath << ": "
                 << file_path << endl;
            f->Close();
            continue;
        }

        f->Close();

        chain->Add(file_path);
        n_added++;
    }

    delete lines;

    return n_added;
}


// ------------------------------------------------------------
// Main plotting function
// ------------------------------------------------------------
//
// This plots hits2.h.channel itself, restricted to a channel range.
// It draws only the overlaid histograms:
//   - no reduced chi-square printing
//   - no chi-square table
//   - no bottom ratio pad
//   - no error bars
//
// Example:
//   plot_channel(3800, 6000)
//   plot_channel(9500, 11276)
//
// ------------------------------------------------------------

void plot_channel(int channel_min = 3800, int channel_max = 6000) {
    gROOT->SetBatch(kTRUE);
    gStyle->SetOptStat(0);

    const char* treePath = "caloskim/TrackCaloSkim";
    const char* branch = "hits2.h.channel";

    int nruns = run_dirs.size();

    if (nruns < 2) {
        cout << "Need at least two run directories." << endl;
        return;
    }

    // --------------------------------------------------------
    // Histogram settings
    // --------------------------------------------------------
    // One bin per channel in the requested range.

    int nbins = channel_max - channel_min;
    double xmin = channel_min;
    double xmax = channel_max;

    if (nbins <= 0) {
        cout << "Invalid channel range: ["
             << channel_min << ", " << channel_max << ")"
             << endl;
        return;
    }

    vector<int> run_numbers;
    vector<TChain*> chains;
    vector<TH1D*> hists;

    // --------------------------------------------------------
    // Channel cut
    // --------------------------------------------------------

    TString channel_cut = "";

    if (use_channel_cut) {
        channel_cut = Form(
            "%s >= %d && %s < %d",
            branch,
            channel_min,
            branch,
            channel_max
        );
    }

    cout << "Using channel cut: " << channel_cut << endl;

    // --------------------------------------------------------
    // Build chains and histograms
    // --------------------------------------------------------

    for (int i = 0; i < nruns; i++) {
        const char* dir = run_dirs[i];
        int run = extract_run_number(dir);
        run_numbers.push_back(run);

        TChain* chain = new TChain(treePath);

        int nfiles = add_good_files_to_chain(chain, dir, treePath);

        // ROOT I/O cache
        chain->SetCacheSize(100 * 1024 * 1024);
        chain->AddBranchToCache(branch, kTRUE);

        cout << "Run " << run << ": added " << nfiles << " ROOT files" << endl;

        if (nfiles == 0) {
            cout << "No DQMValidationTrees ROOT files found in directory: " << dir << endl;
            return;
        }

        cout << "Run " << run << " entries = " << chain->GetEntries() << endl;

        chains.push_back(chain);

        TH1D* h = new TH1D(
            Form("h_channel_run_%d_ch_%d_%d", run, channel_min, channel_max),
            Form(
                "Hit Channel Comparison, channels [%d,%d);Channel;Hits",
                channel_min,
                channel_max
            ),
            nbins,
            xmin,
            xmax
        );

        // Do not call Sumw2 and do not draw error bars.

        chain->Draw(
            Form("%s >> h_channel_run_%d_ch_%d_%d", branch, run, channel_min, channel_max),
            channel_cut,
            "goff"
        );

        cout << "Raw number of hits in channel histogram for run "
             << run
             << " channels [" << channel_min << ", " << channel_max << ")"
             << " = "
             << h->Integral()
             << endl;

        hists.push_back(h);
    }

    // --------------------------------------------------------
    // Normalize all non-reference runs to the reference run
    // --------------------------------------------------------

    TH1D* h_ref = hists[0];
    int run_ref = run_numbers[0];

    double ref_integral = h_ref->Integral();

    for (int i = 1; i < nruns; i++) {
        double integral = hists[i]->Integral();

        if (integral > 0) {
            double scale = ref_integral / integral;
            hists[i]->Scale(scale);

            cout << "Scale factor applied to run "
                 << run_numbers[i]
                 << " = "
                 << scale
                 << endl;
        }
    }

    // --------------------------------------------------------
    // Colors and styles
    // --------------------------------------------------------

    vector<int> colors = {
        kBlack,
        kRed,
        kBlue,
        kGreen + 2,
        kOrange + 7,
        kMagenta,
        kCyan + 2,
        kViolet,
        kBrown
    };

    for (int i = 0; i < nruns; i++) {
        int color;

        if (color_by_good_bad) {
            color = get_run_color(run_numbers[i]);
        } else {
            color = colors[i % colors.size()];
        }

        hists[i]->SetLineColor(color);
        hists[i]->SetLineWidth(2);
        hists[i]->SetMarkerColor(color);
    }

    // --------------------------------------------------------
    // Canvas: single pad only
    // --------------------------------------------------------

    TCanvas* c = new TCanvas(
        Form("c_channel_ch_%d_%d", channel_min, channel_max),
        "channel comparison",
        1200,
        700
    );

    c->SetGridx();
    c->SetGridy();

    // --------------------------------------------------------
    // Main plot only
    // --------------------------------------------------------

    double ymax = 0.0;

    for (int j = 0; j < nruns; j++) {
        for (int i = 1; i <= nbins; i++) {
            ymax = TMath::Max(ymax, hists[j]->GetBinContent(i));
        }
    }

    hists[0]->SetTitle(
        Form(
            "Hit Channel Comparison, channels [%d,%d);Channel;Hits",
            channel_min,
            channel_max
        )
    );

    hists[0]->SetMaximum(1.15 * ymax);
    hists[0]->SetMinimum(0);

    hists[0]->GetXaxis()->SetTitle("Channel");
    hists[0]->GetYaxis()->SetTitle("Hits");
    hists[0]->GetXaxis()->SetTitleSize(0.045);
    hists[0]->GetYaxis()->SetTitleSize(0.045);
    hists[0]->GetXaxis()->SetLabelSize(0.040);
    hists[0]->GetYaxis()->SetLabelSize(0.040);

    hists[0]->Draw("HIST");

    for (int i = 1; i < nruns; i++) {
        hists[i]->Draw("HIST SAME");
    }

    gPad->RedrawAxis();

    TLegend* leg = new TLegend(0.56, 0.55, 0.88, 0.88);
    leg->SetBorderSize(1);
    leg->SetFillColor(kWhite);

    for (int i = 0; i < nruns; i++) {
        if (i == 0) {
            leg->AddEntry(
                hists[i],
                Form(
                    "Data ID %d reference (%s)",
                    run_numbers[i],
                    get_run_status(run_numbers[i])
                ),
                "l"
            );
        } else {
            leg->AddEntry(
                hists[i],
                Form(
                    "Data ID %d normalized (%s)",
                    run_numbers[i],
                    get_run_status(run_numbers[i])
                ),
                "l"
            );
        }
    }

    leg->Draw();

    // --------------------------------------------------------
    // Save output
    // --------------------------------------------------------

    TString run_tag = "";

    for (int i = 0; i < nruns; i++) {
        if (i > 0) run_tag += "_";
        run_tag += Form("%d", run_numbers[i]);
    }

    TString mode_tag = color_by_good_bad ? "good_bad_colors" : "multi_colors";

    TString png_name = Form(
        "channel_comparison_nochi2_nobottom_noerrors_%s_ch_%d_%d_%s.png",
        mode_tag.Data(),
        channel_min,
        channel_max,
        run_tag.Data()
    );

    TString pdf_name = Form(
        "channel_comparison_nochi2_nobottom_noerrors_%s_ch_%d_%d_%s.pdf",
        mode_tag.Data(),
        channel_min,
        channel_max,
        run_tag.Data()
    );

    c->SaveAs(png_name);
    c->SaveAs(pdf_name);

    cout << "Saved " << png_name << " and " << pdf_name << endl;
}


// ------------------------------------------------------------
// Plot all predefined channel ranges
// ------------------------------------------------------------

void plot_all_channel_ranges() {
    for (auto range : channel_ranges) {
        int ch_min = range.first;
        int ch_max = range.second;

        cout << endl;
        cout << "========================================" << endl;
        cout << "Plotting channel histogram range ["
             << ch_min
             << ", "
             << ch_max
             << ")"
             << endl;
        cout << "========================================" << endl;

        plot_channel(ch_min, ch_max);
    }
}


// ------------------------------------------------------------
// Wrapper for file name plot_channelhist.C
// ------------------------------------------------------------
// If this file is named plot_channelhist.C, then this command works:
//   root -l -q -b plot_channelhist.C
// ------------------------------------------------------------

void plot_channelhist() {
    plot_all_channel_ranges();
}

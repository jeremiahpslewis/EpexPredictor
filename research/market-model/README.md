# Market model of the German day-ahead auction: research scripts

The scripts and the running notes behind `predictor/model/marketmodel.py`. They are kept so that the work can be
checked and so that the frozen forward test can be scored later. They are not part of the application: nothing
here is imported by it, nothing is installed with it, and the Docker image does not contain this folder.

## What is here

- `notes/coupled_market_notes_2.txt` is the record: every model level, what was expected before each fit or test,
  what came out, and every correction. Sections are numbered 21.x; the later ones supersede the earlier ones where
  they differ. `notes/coupled_market_notes.txt` is the short earlier file.
- `exp/` holds the scripts as they were run. The ones the final result rests on:
  - `market1.py`, `market1n.py`: the market model and its fit (prior predictive, simulate and recover, fit).
  - `shift_build.py`: the supply shift for every known quarter-hour price.
  - `ts_gen0.py`, `ts_nbm.py`: the load, wind and solar forecast for Germany and the neighbours' residual loads.
  - `ts_shift.py`: the forecast of the supply shift. `ts_live3m.py`: the forecast of the price itself.
  - `sealed_score.py`: the scoring of the sealed days, written and hashed before their forecasts existed.
  - `day_dist.py`: the quantiles of both routes and their average.
- `forward/` holds two data-building commands that were only ever run inline and were recovered afterwards.

## What is not here

No data, no model weights, no credentials. The scripts read an ENTSO-E key from a file that is not in the
repository, and day-ahead prices and the other inputs have to be downloaded again.

## Before running anything

The scripts were written in a temporary workspace and carry its absolute path in a variable called `SP`. That
path has to be changed, and the data files it expects have to be rebuilt first. About ten downloaders feed them;
section 21.97 of the notes lists each file and the script that builds it. The scripts are terse and were not
written to be maintained.

## The frozen forward test

Section 21.97 of the notes and the section "Frozen specification for the forward test" of the review document fix
what is to be scored on days after 9 November 2026: the average of the two routes as the forecast, the comparisons,
and one reading once at least 30 scored days exist. The hashes given there are of the files in `exp/`.

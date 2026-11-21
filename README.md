# EPEX day-ahead price prediction

This is a simple statistical model to predict EPEX day-ahead prices based on various parameters.
It works to a reasonably good degree; see [Model performance](#model-performance) for how it was measured.
This repository includes
- The self-training prediction model itself
- A simple FastAPI app to get a REST API up
- A Docker compose file to have it running wherever

Supported Countries:
- Germany (default)
- Austria
- France
- Belgium
- Netherlands
- Sweden (SE1-SE4)
- Denmark (DK1-DK2)
- Spain
- Portugal
- Others can be added relatively easily, if there is interest


## Lookout
- Maybe package it directly as a Home Assistant Add-on

## The Model
We sample multiple locations distributed across each region. We fetch [Weather data from Open-Meteo.com](https://open-meteo.com/) for those locations for the past n days (default n=540).
This serves as the main data source. For Germany, a few additional locations in the North Sea are sampled, too.

Price data is provided under CC BY 4.0 by smartd.de, retrieved via [api.energy-charts.info](https://api.energy-charts.info/) and [ENTSO-E transparency platform](https://transparency.entsoe.eu/). Optionally (see configuration below), prices can also be fetched from the [Nord Pool data portal API](https://dataportal-api.nordpoolgroup.com/api/DayAheadPrices) as a lowest-priority fallback.

Grid load data is provided by [ENTSO-E transparency platform](https://transparency.entsoe.eu/).

For Germany, the day-ahead forecasts of wind power, solar power and load of the transmission system operators are retrieved via [api.energy-charts.info](https://api.energy-charts.info/) (`public_power_forecast`), and hourly weather on a dense grid of 58 locations from [Open-Meteo.com](https://open-meteo.com/).

Natural gas prices are provided by the open [data export of Trading Hub Europe](https://api.tradinghub.eu/api/dataexport/manual/en), the German gas market area manager.

### Features

Weather features (per sample location):
- Wind speed at 80m
- Temperature at 2m
- Global tilted irradiance (solar)
- Air pressure at mean sea level
- Relative humidity

Time features:
- Azimuth of the sun as indicator of time of day
- Elevation of the sun
- Day of the week (Monday to Sunday)
- Holiday indicator (regional holidays weighted by fraction of regions, e.g. 0.5 if half the regions have the holiday)
- Slot of the day in local time (0-95, one per 15 minute interval)
- Unix timestamp, lets the model pick up slow drift within its training window
- Sunrise influence: how many seconds between sunrise and the current time slot
- Sunset influence: how many seconds between sunset and the current time slot
  (a sentinel value is used on dates without sunrise/sunset, i.e. polar day/night)

Other:
- Entso-E load forecast (optional, but highly recommended, especially for DE and AT)
- Natural gas price (daily average at the German virtual trading point), forward filled (select regions only)
- Stage-1 price forecasts of all regions (stage-2 models only, see below)
- Europe-wide weather: wind power potential, irradiance and temperature averaged over several zones, plus the wind power potential around the North Sea
- Residual load (Germany only, see below)

Output:
- Electricity price

## How it works
The model uses **LightGBM gradient boosting** to predict electricity prices. LightGBM automatically learns non-linear relationships and feature interactions, making it well-suited for electricity price prediction where factors like low wind+solar can cause price spikes due to merit order pricing.

The prediction runs in **two stages**:
1. **Stage 1**: one base model per region, trained on that region's own features (weather, prices, load, gas).
2. **Stage 2**: one model per region, trained on the same features plus the **stage-1 forecasts of all regions** as extra input features.

The cross-region forecasts let the model pick up on price coupling between neighboring markets (e.g. a wind lull in one region pushing up prices in connected ones), which a single-region model cannot see.

For regions that use the natural gas price, each stage consists of **two models**: one predicts the price itself, the other one predicts **price / natural gas price**, which is multiplied by the latest known gas price again. The forecast is the average of both. Whenever gas power plants set the price, electricity prices follow the gas price almost proportionally, and a tree-based model cannot predict a price level it has never seen during training - the second model can, and it keeps older training data comparable when gas prices move. But gas does not set the price all the time, so neither model is right on its own.

### Residual load (Germany)
The day-ahead auction clears on what the market expects the **residual load** to be: load minus wind and solar power. The transmission system operators publish that expectation as day-ahead forecasts, but one day ahead only, which is too late for a price forecast. So three more LightGBM models learn the published wind power, solar power and load from the weather on a dense grid of locations across Germany and at its offshore wind farms. The price model gets the published residual load where there is one (all of its training data), and the forecast of these models for the days ahead. The few sample locations of the weather features above are too coarse for this: wind power follows the wind at the turbines.

If the published forecasts or the grid weather cannot be fetched, the price model is trained without this input. Times for which only this input is missing (for example when the grid weather could not be updated) are forecast by a second set of models trained without it.

### Timestral (optional, Germany)
[Timestral](https://github.com/mistralai/timestral) is a pretrained time series model by Mistral. It forecasts the price from its recent course plus the weather forecast and the residual load, without any training, and makes different mistakes than the tree models: it is best right after the last known price. When it is switched on (see configuration below), its forecast is averaged with the one of the tree models for the first 38 hours after the last known price, fading out until 48 hours. It needs PyTorch, about 2 GB more disk space and 1 GB more memory (with the market model and the residual load forecast below, the service peaked at 3.4 GB with eight regions and evaluation plots requested, so 4 GB is enough; an ENTSO-E key adds data and was not measured), and its weights are downloaded from Hugging Face on first use. The residual load as input was backtested with a better residual load forecast than the one of this application, where it lowered Timestral's error by day considerably; with this application's forecast it is not backtested yet, and `EPEXPREDICTOR_TIMESTRAL_RESIDUAL_LOAD=false` leaves it out. The eval plot shows the tree models only.

### Market model (with Timestral, Germany)
A model of the German auction itself: supply by block (lignite, hard coal and gas plant with their costs from the gas price, net imports, pumped storage) against the price, and the price at which it meets the residual load. On its own it forecasts poorly. What makes it useful is the supply shift: how much more or less supply there was than the model says, in GW, worked out from every known price. Timestral continues that series, and the market is cleared again with the forecast shift and the forecast residual load.

It runs whenever Timestral is switched on, with the same model, and counts as a third forecaster: tree models, Timestral on the price and the market model get a third each for the first 38 hours after the last known price, fading out until 48 hours. If one of them cannot deliver, the others share the weight. `EPEXPREDICTOR_MARKET_MODEL=false` switches it off. It needs a second Timestral run per update and nothing else: no further data is fetched (the carbon price and the output of biomass, hydro and the like are constants in it, the neighbours are left out, and the gas price is the one of two days before).

What to expect: in backtests of the next day's quarter hours from midnight, the market model and Timestral on the price were equally good, and their average was better than Timestral alone on days that had been set aside beforehand (by 6 % over the whole day). Added to the tree models and Timestral, it lowered their error over the whole day by 7 to 13 % when it was given a good residual load forecast. With the residual load forecast of this application the gain was smaller: 4 % with the weather forecast of the day before, none with a weather forecast two to three days old. So expect a small improvement close to the last known price and none further out. The scripts and notes of those backtests are in [`research/market-model`](research/market-model).

### Residual load from Timestral (with Timestral, Germany)
The grid operators publish their forecasts of wind power, solar power and load one day ahead. For the time after that, the tree models of the residual load read the three figures off the weather forecast. With Timestral switched on, Timestral also continues the three published series, with the same weather forecast as input. It starts from the last published level, which the tree models do not know. Right after the last published figure that makes it the better forecast; later neither is better, and the two make different mistakes. So they are mixed: Timestral alone at the last published figure, half and half from a day later on, for two and a half days in all.

What the backtests showed (forecasts made at midnight for the day that follows; 146 days in winter, summer and autumn that had been used for other tests before):

- Residual load: the mix had a lower error than the tree models in every season, by 16 to 49 % in the first six hours and by 9 to 32 % over the first day. The large figures are from winter and summer, where the tree models were given an older weather forecast than they have in live operation.
- Price, autumn, against the tree models' residual load: the mix of tree models, Timestral and market model had a 2 % lower error over the whole day with it (not a clear difference; a plain average of the two residual load forecasts gave 3 %).
- Price, against a plain average of the two residual load forecasts: 8 % lower error at night in winter and 4 % in summer, no clear change in autumn or over the whole day. That is why Timestral gets the larger weight at the start.

How much weight the tree models deserve depends on how old their weather forecast is, and the backtests could only approximate that. The log below records both forecasts, so the weighting can be checked on live data.

Timestral on the price and the market model use this forecast; the tree models of the price keep the residual load of the tree models. It costs a third Timestral run whenever the published figures or the weather forecast change (about 20 seconds on four CPU cores). `EPEXPREDICTOR_TIMESTRAL_LOAD=false` switches it off.

### Forecast log
With Timestral switched on and a data directory set, every forecast that is issued for Germany is also written to `forecastlog_DE_LU.csv.gz` in the data directory: for the two days after the last known price, what the tree models, Timestral and the market model each said and what was issued, once per hour at most. With the prices that follow, this shows how good each of them was. The file is only ever added to (about 30 MB a year). The two forecasts of the residual load are written to `loadforecastlog_DE_LU.csv.gz` in the same way, for the time after the last published figure. A third file, `coreforecastlog_DE_LU.csv.gz`, holds the core of the market model next to the tree models and the market model: the supply curves with one level of supply shift, the mean of the last seven days, and no Timestral. It is not part of the issued forecast. Over 2.7 years of data this core, fitted once, kept its accuracy relative to the price level (an error of 13 to 22 % of the mean price in every quarter) while the price level doubled; the log shows whether that holds, and how much Timestral adds to it (on the backtest days the market model's error was a fifth to a third lower than the core's).

`python -m predictor.scorelog <data directory>` scores both logs against the prices and published figures that followed: the mean absolute error of each forecaster in the first six hours, the first day and the second day after the last known value, and its difference to the tree models with a range from resampling days. With docker: `docker compose exec epexpredictor python -m predictor.scorelog /data`. A few weeks of log are needed before the ranges mean much.

A time for which one of the inputs is missing, for example because the weather service could not be reached for several days, gets no forecast, so the output of the API can end earlier than usual. The load forecast is the exception: where it is missing, a forecast is still made.

## Model performance
There are two separate kinds of evaluation. Their figures are not comparable with each other.

### Rolling backtest with historical weather
`predictor/performance_testing.py` runs a rolling backtest of the full two-stage stack. Stage-1 models are trained for all regions, then stage-2 models are trained with the stage-1 forecasts of all regions as cross features. Set `TWO_STAGE = False` to fall back to the single-stage evaluation. Run it from the repository root:

```bash
EPEXPREDICTOR_DATADIR=./data EPEXPREDICTOR_ENTSOE_API_KEY=... python predictor/performance_testing.py
```

The configuration is set by the constants at the top of the script: `START` 2025-09-01 and `END` 2026-09-01 (UTC), `LEARN_DAYS` 540, all 13 regions in `CROSS_REGIONS` and `EVAL_REGIONS`, two-stage. One window starts at each UTC midnight from `START` to four days before `END`, so that all three forecast days end by `END`. Each window is trained on the `LEARN_DAYS` before its start, with every data store cut off at the start, so no later price is used. It then forecasts three forecast days. Day k is the 24 hours [start + k − 1 days, start + k days) in UTC. The intervals are half-open, so every quarter hour belongs to exactly one day. The weather is the historical weather, not the forecast that was available at the time, so live errors can be larger.

For each region and forecast day the script prints:
- **Expected:** the quarter hours of that day over all windows.
- **Actual:** those with an actual price.
- **Covered:** for each model, those with an actual price and a forecast of that model, with the share of Actual. A model gives no forecast where a required input is missing, so coverage below 100 % means prices that were not issued.
- **Common:** those with an actual price and a forecast of every model.
- **MAE and RMSE** of stage 1, stage 2 and the same-slot-last-week baseline, on the Common quarter hours only, so all three are compared on the same timestamps. The baseline is the actual price 7 × 24 hours earlier, which is known when the window starts. Errors are pooled over the quarter hours of all windows.
- **MAE relative to the baseline:** below 1 means better than last week's price.

Prices and errors are in ct/kWh (EUR/MWh ÷ 10).

The table for the current configuration has not been generated yet. Running it is several hours of CPU time per region.

The table below is from an earlier version and is kept for reference only:
- **Configuration:** 180-day training window, no gas-relative model, no Europe-wide weather features, two-stage, data 2025-09-01 to 2026-09-01, historical weather.
- **Metric:** the old one. The quarter hour at each day boundary was counted in two days, errors were averaged per window rather than pooled, and there was no coverage count and no baseline.
- **Not included:** the residual load input and Timestral.

Results, ct/kWh (1-, 2- and 3-day-ahead):
| Region | 1d RMSE | 1d MAE | 2d RMSE | 2d MAE | 3d RMSE | 3d MAE |
|--------|---------|--------|---------|--------|---------|--------|
| DE     | 2.87    | 1.62   | 3.17    | 1.84   | 3.26    | 1.92   |
| AT     | 2.95    | 1.78   | 3.31    | 2.1    | 3.41    | 2.21   |
| BE     | 3.09    | 1.69   | 3.48    | 1.97   | 3.56    | 2.05   |
| NL     | 3.02    | 1.68   | 3.37    | 1.92   | 3.5     | 2.02   |
| SE1    | 2.35    | 1.46   | 3.03    | 1.95   | 3.25    | 2.13   |
| SE2    | 2.3     | 1.4    | 2.9     | 1.84   | 3.12    | 2.01   |
| SE3    | 2.6     | 1.76   | 3.03    | 2.16   | 3.19    | 2.29   |
| SE4    | 2.97    | 2.02   | 3.48    | 2.47   | 3.68    | 2.63   |
| DK1    | 2.62    | 1.64   | 3.03    | 1.92   | 3.14    | 2.02   |
| DK2    | 3.02    | 1.84   | 3.35    | 2.16   | 3.45    | 2.24   |
| ES     | 1.99    | 1.39   | 2.4     | 1.73   | 2.55    | 1.84   |
| PT     | 1.97    | 1.38   | 2.43    | 1.75   | 2.6     | 1.87   |
| FR     | 2.69    | 1.75   | 3.15    | 2.15   | 3.31    | 2.27   |

#### Residual load input and Timestral (Germany)
These were backtested separately:
- forecasts made at 14:00 for the day after tomorrow;
- the weather as it was forecast at that time, at least 48 hours ahead;
- mean absolute error in ct/kWh of the first 14 hours of that day;
- autumn 2026 (66 forecasts) / spring–summer 2026 (50 forecasts).

| | Whole 14 hours | 00:00–08:00 only |
|---|---|---|
| Without them | 1.86 / 1.50 | 1.65 / 1.06 |
| With the residual load input | 1.72 / 1.30 | 1.49 / 0.87 |
| With Timestral-123M on top | 1.59 / 1.30 | 1.30 / 0.80 |

Timestral on its own was not better than the tree models.

### Forecasts as issued (Germany)
`python -m predictor.scorelog <data directory>` scores the forecast log described above. These are the forecasts the service actually issued, with the weather forecast of the time. It reports the following, in ct/kWh, for the first six hours, the first day and the second day after the last known price:
- the issues and days of issue logged;
- the quarter hours due, those with an actual price, and those each forecaster covered;
- the mean absolute error of the tree models, Timestral, the market model, the issued blend (`price`) and the same-slot-last-week baseline (`last_week`).

With fewer than 28 days of issue the figures are printed with the conclusion left open. No result from the log is published here yet.

Some observations:
- At night, predictions are typically within 0.5 ct/kWh
- Morning/Evening peaks are typically within 1-1.5 ct/kWh
- Extreme peaks due to "Dunkelflaute" are correctly detected, but estimation of the exact price is a challenge (e.g. the model might predict 75ct while reality is 60ct or vice versa)
- High PV noons are usually correctly detected with good accuracy


### Current forecast (DE)
![image](https://epexpredictor.batzill.com/eval_plot?region=DE&transparent=false&width=1024&height=512)


Feel free to generate your own plot for other time ranges or regions [here](https://epexpredictor.batzill.com/docs#/default/generate_evaluation_plot_eval_plot_get).

Note that the eval plot trains the full 2-stage model stack (all regions) for the requested historical window, which is CPU-intensive. Results are therefore cached for 30 minutes per (region, range) - after that, the last result is served while a refresh runs in the background.


# Public API
You can find a freely accessible installment of this software [here](https://epexpredictor.batzill.com/).
Get a glimpse of the current prediction [here](https://epexpredictor.batzill.com/prices).

There are no guarantees given whatsoever - it might work for you or not.
I might stop or block this service at any time. Fair use is expected!

# Self Hosting
You can easily self-host this software. For easy deployment, check out the docker compose file.
You will probably want to register with Entso-E and request an API key.
Without Entso-E API access
- some parameters are missing and the model will perform significantly worse, especially for DE and AT
- Some regions will not be available (e.g. SE1-4)

### Configuration (environment variables)

| Variable                        | Default      | Description |
|---------------------------------|--------------|-------------|
| `EPEXPREDICTOR_DATADIR`         | *(unset)*    | Directory where the app persists its cached data. Unset by default (no persistence - the app re-fetches everything on startup). In the compose file this is set to the mounted `/data/` volume. |
| `EPEXPREDICTOR_ENTSOE_API_KEY`  | *(unset)*    | Entso-E API key. Highly recommended: it acts as a fallback price source, is required for some regions (e.g. SE1-SE4), and provides the electrical load forecast which significantly improves model performance. |
| `EPEXPREDICTOR_TIMESTRAL_MODEL` | *(unset)*    | Name of a Timestral model on Hugging Face, e.g. `mistralai/Timestral-123M`, to average its forecast with the one of the tree models for Germany. Needs the packages of `requirements-timestral.txt` with the CPU build of PyTorch (`uv pip install --torch-backend cpu -r requirements-timestral.txt`, or see the file for pip). The model weights have their own license, see the model page. |
| `EPEXPREDICTOR_MARKET_MODEL`     | `true`       | Set to `false` to leave the market model out of the forecast for Germany. It only runs when Timestral is switched on. |
| `EPEXPREDICTOR_TIMESTRAL_RESIDUAL_LOAD` | `true` | Set to `false` to run Timestral on the price without the residual load as input. |
| `EPEXPREDICTOR_TIMESTRAL_LOAD` | `true` | Set to `false` to give Timestral and the market model the residual load forecast of the tree models instead of the one of Timestral. |
| `EPEXPREDICTOR_TIMESTRAL_REVISION` | *(tested revision)* | Revision of the Timestral model weights on Hugging Face. Defaults to the revision the backtests were run with for the published Timestral models, `main` loads the latest. |
| `EPEXPREDICTOR_USE_NORDPOOL`    | *(disabled)* | Set to `true` (also `1`/`yes`/`on`) to additionally use the [Nord Pool data portal API](https://dataportal-api.nordpoolgroup.com/api/DayAheadPrices) as a price source. It is the lowest-priority source and only queried if energy-charts and ENTSO-E don't deliver. Nord Pool data is only available for ~4 weeks of history. **Note:** enabling this is up to you - make sure your use case complies with the Nord Pool terms of use / data license. |

### Proxmox VE (LXC container)

The two scripts in [`proxmox/`](proxmox/) create an LXC container on a Proxmox VE host and install EpexPredictor in it as a systemd service. They are written like the [community-scripts](https://community-scripts.org) and run on their engine, so the usual dialog with default and advanced settings appears. Run this in the shell of the Proxmox host:

```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/jeremiahpslewis/EpexPredictor/main/proxmox/ct/epexpredictor.sh)"
```

The container gets 2 cores, 4 GB of memory and 6 GB of disk by default, and the API listens on port 8000. It answers once the first download of price and weather data has finished, which takes a quarter of an hour or more. `update` in the shell of the container pulls the latest code.

Settings go into `/opt/epexpredictor/.env`, one `NAME=value` per line. For the ENTSO-E API key, open a shell in the container (`pct enter <id>` on the host, or the console in the web interface) and run:

```bash
nano /opt/epexpredictor/.env      # remove the # in front of EPEXPREDICTOR_ENTSOE_API_KEY= and put the key after the =
systemctl restart epexpredictor
```

To try another branch of the repository, for example of a pull request, run in the container (`--timestral` also installs and switches on Timestral; running it again updates, and `main` switches back):

```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/jeremiahpslewis/EpexPredictor/<branch>/proxmox/switch-branch.sh)" _ <branch> --timestral
```

For Timestral by hand, also in the container:

```bash
cd /opt/epexpredictor
uv pip install --torch-backend cpu -r requirements-timestral.txt
echo "EPEXPREDICTOR_TIMESTRAL_MODEL=mistralai/Timestral-123M" >> .env
systemctl restart epexpredictor
```

# Home Assistant integration
At some point, I might create a HA addon to run everything locally.
For now, you have to either use my server, or run it yourself.

Note: Home Assistant only supports a limited amount of data in state attributes. Therefore, we use the "short format" output, and limit the time to 120 hours.
If you need more, you will have to be more creative.
Personally, I provide the data as a HA "service" (now "action") using pyscript, and then call this service to work with the data.



### Configuration:
```yaml
# Make sure you change the parameters region, surcharge and taxPercent according to your electricity plan
sensor:
  - platform: rest
    resource: "https://epexpredictor.batzill.com/prices_short?region=DE&surcharge=13.70084&taxPercent=19&unit=EUR_PER_KWH&hours=120"
    method: GET
    unique_id: epex_price_prediction
    name: "EPEX Price Prediction"
    unit_of_measurement: €/kWh
    value_template: "{{ value_json.t[0] }}"
    json_attributes:
      - s
      - t

  # If you want to evaluate performance in real time, you can add another sensor like this
  # and plot it in the same diagram as the actual prediction sensor

  #- platform: rest
  #  resource: "https://epexpredictor.batzill.com/prices_short?region=DE&surcharge=13.70084&taxPercent=19&evaluation=true&unit=EUR_PER_KWH&hours=120"
  #  method: GET
  #  unique_id: epex_price_prediction_evaluation
  #  name: "EPEX Price Prediction Evaluation"
  #  unit_of_measurement: €/kWh
  #  value_template: "{{ value_json.t[0] }}"
  #  json_attributes:
  #    - s
  #    - t
```

### Display, e.g. via Plotly Graph Card:
```yaml
type: custom:plotly-graph
time_offset: 26h
layout:
  yaxis9:
    fixedrange: true
    visible: false
    minallowed: 0
    maxallowed: 1
entities:
  - entity: sensor.epex_price_prediction
    name: EPEX Price Prediction
    unit_of_measurement: ct/kWh
    texttemplate: "%{y:.0f}"
    mode: lines+text
    textposition: top right
    filters:
      - fn: |-
          ({xs, ys, meta}) => {
            return {
              xs: xs.concat(meta.s.map(s => s*1000)),
              ys: ys.concat(meta.t).map(t => +t*100)
            }
          }
  - entity: ""
    name: Now
    yaxis: y9
    showlegend: false
    line:
      width: 1
      dash: dot
      color: orange
    x: $ex [Date.now(), Date.now()]
    "y":
      - 0
      - 1
hours_to_show: 30
refresh_interval: 10
```

# evcc integration

[evcc](https://evcc.io/) is an open-source EV charging controller that can optimize charging based on electricity prices.
It now has native support for EpexPredictor, see [docs](https://docs.evcc.io/docs/tariffs#epex-predictor-predicted-epex-spot-prices)
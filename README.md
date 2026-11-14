# EPEX day-ahead price prediction

This is a simple statistical model to predict EPEX day-ahead prices based on various parameters.
It works to a reasonably good degree. Better than many of the commercial solutions.
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

If the published forecasts or the grid weather cannot be fetched, the price model is trained without this input.

### Timestral (optional, Germany)
[Timestral](https://github.com/mistralai/timestral) is a pretrained time series model by Mistral. It forecasts the price from its recent course plus the weather forecast, without any training, and makes different mistakes than the tree models: it is best right after the last known price. When it is switched on (see configuration below), its forecast is averaged with the one of the tree models for the first 38 hours after the last known price, fading out until 48 hours. It needs PyTorch, about 2 GB more disk space and 1 GB more memory, and its weights are downloaded from Hugging Face on first use. The eval plot shows the tree models only.

A time for which one of the inputs is missing, for example because the weather service could not be reached for several days, gets no forecast, so the output of the API can end earlier than usual. The load forecast is the exception: where it is missing, a forecast is still made.

## Model performance
For performance testing, see `predictor/performance_testing.py`. It runs a rolling backtest of the full 2-stage stack: stage-1 models for all regions, then stage-2 models using the stage-1 forecasts of all regions as cross-features (set `TWO_STAGE = False` to fall back to the original single-stage evaluation).

Remarks:
- Tests were run in 2026, with data from 2025-09-01 to 2026-09-01. The model is tuned for 15 minute pricing.
- The model uses a 540-day rolling training window
- The table below was measured with the previous setup (180-day training window, no gas-relative model, no Europe-wide weather features) and still has to be regenerated
- Tests were done with historical weather data. If the weather forecast is wrong, performance might be slightly worse in practice
- The residual load input and Timestral are not part of the table either. In a backtest for Germany over ten weeks (66 forecasts made at 14:00 for the day after tomorrow, with the weather as it was forecast at that time), the mean absolute error of the first 14 hours of that day was 1.86 ct/kWh without them, 1.52 to 1.72 with the residual load input (depending on the age of the weather forecast), and 1.48 to 1.59 with Timestral-123M on top

Results (1/2/3-day ahead prediction):
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
| `EPEXPREDICTOR_TIMESTRAL_MODEL` | *(unset)*    | Name of a Timestral model on Hugging Face, e.g. `mistralai/Timestral-123M`, to average its forecast with the one of the tree models for Germany. Needs the packages of `requirements-timestral.txt` (`pip install -r requirements-timestral.txt`). The model weights have their own license, see the model page. |
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

For Timestral, also in the container:

```bash
cd /opt/epexpredictor
uv pip install -r requirements-timestral.txt
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
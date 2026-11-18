"""
A market model of the German day-ahead auction: supply by block against the price, and the price that clears it.

Supply is the sum of lignite, hard coal and gas plant (each with its own cost from the gas and the carbon price),
net imports, pumped storage and a constant for generation that is in no measured category. Demand is the residual
load less the other plant (biomass, hydro, waste, ...), and does not react to the price. Each quarter hour is its
own market.

The curves alone forecast the price poorly. What makes them useful is the supply shift: how much more or less
supply there was than the curves say, in GW, worked out from every known price with `supply_shift`. That one
number per quarter hour is what a forecaster continues into the future, and `clearing_price` turns its forecast
back into a price. Uncertainty stated in GW becomes a price range the same way, because the clearing price falls
steadily as the shift grows.

The parameters are the medians of a Bayesian fit to hourly German data from March 2024 to November 2025 (price,
output by fuel, net imports, storage, published residual loads of Germany and its neighbours). In backtests of the
next day's quarter-hour prices from midnight, this model with its shift forecast by Timestral was level with
Timestral forecasting the price directly, and the average of the two was better than Timestral alone over the
whole day.

This module holds the model only. It is not used by the forecast yet: that needs the output of the other plant,
the carbon price and the neighbours' forecasts as inputs, which the application does not fetch so far.
"""

import numpy as np
from scipy.special import ndtr

# Installed capacity per fuel in GW, as assumed in the fit
CAPACITY = {"lignite": 18.0, "coal": 12.0, "gas": 33.0}

# Prices within this distance of zero count as zero (EUR/MWh)
ZERO_BAND = 0.1

# Residual load forecasts the import block reads, in this order, with the mean and spread they were standardised with (GW)
NEIGHBOURS = ("FR", "NL", "BE", "AT", "PL", "CZ", "CH", "DK")
NEIGHBOUR_MEAN = np.array([39.0209, 8.3987, 6.6042, 4.9794, 13.0607, 6.3852, 6.3631, 1.7613])
NEIGHBOUR_SPREAD = np.array([10.4142, 3.0695, 2.0073, 1.6687, 4.2313, 1.2438, 1.3074, 1.2803])
# GW of net imports per standard deviation of each neighbour's residual load
NEIGHBOUR_EFFECT = np.array([-0.06905, -0.41101, -1.61489, -0.96507, -0.38269, -1.10395, -0.62102, 0.01636])

PARAMETERS = {
    # cost in EUR/MWh: lignite and hard coal a constant plus their carbon, gas a multiple of the cost index
    "lignite_cost": 9.174429, "coal_cost": 48.482555, "gas_cost_factor": 1.374758,
    # per fuel: the share that runs at any price, the share of the rest that responds to the price,
    # and the width of the range of price over cost in which it comes on
    "lignite_mustrun": 0.184696, "lignite_responsive": 0.549972, "lignite_width": 0.38573,
    "coal_mustrun": 0.069452, "coal_responsive": 0.485287, "coal_width": 0.317356,
    "gas_mustrun": 0.047115, "gas_responsive": 0.384009, "gas_width": 0.383722,
    # net imports in GW: at price = cost index, per unit of price over cost index, and the step down near zero price
    "import_level": 5.423048, "import_slope": 15.109554,
    "import_step": 6.762943, "import_step_price": -0.887913, "import_step_width": 2.982941,
    # pumped storage: from pumping to generating around a price over cost index
    "storage_range": 4.319945, "storage_centre": 1.000168, "storage_width": 0.36189,
    # generation in no measured category, and the plant that stops below zero price (GW)
    "unmeasured": 3.449766, "stops_below_zero": 1.390832,
}

COAL_CARBON = 0.85  # tonnes of carbon per MWh of hard coal relative to lignite's one


def cost_index(gas_price, carbon_price):
    """Cost of power from gas including its carbon, EUR/MWh (gas in EUR/MWh, carbon in EUR/t)"""
    return 1.85 * np.asarray(gas_price, dtype=float) + 0.37 * np.asarray(carbon_price, dtype=float)


def neighbour_imports(residual_loads) -> np.ndarray:
    """
    What the neighbours' residual loads add to Germany's net imports, in GW.

    residual_loads: array (..., 8), the published forecasts in GW in the order of NEIGHBOURS
    """
    standardised = (np.asarray(residual_loads, dtype=float) - NEIGHBOUR_MEAN) / NEIGHBOUR_SPREAD
    return standardised @ NEIGHBOUR_EFFECT


def supply(price, gas_price, carbon_price, parameters: dict = PARAMETERS) -> np.ndarray:
    """
    Supply in GW at a price in EUR/MWh, without the neighbours' effect and without the plant that stops below zero.
    All arguments broadcast against each other.
    """
    p = np.asarray(price, dtype=float)
    carbon = np.asarray(carbon_price, dtype=float)
    index = cost_index(gas_price, carbon)
    fuel_cost = {
        "lignite": parameters["lignite_cost"] + carbon,
        "coal": parameters["coal_cost"] + COAL_CARBON * carbon,
        "gas": parameters["gas_cost_factor"] * index,
    }
    total = np.zeros(np.broadcast(p, index).shape)
    for fuel, capacity in CAPACITY.items():
        mustrun = parameters[f"{fuel}_mustrun"]
        coming_on = ndtr((p / fuel_cost[fuel] - 1.0) / parameters[f"{fuel}_width"])
        total = total + capacity * (mustrun + (1.0 - mustrun) * parameters[f"{fuel}_responsive"] * coming_on)

    ratio = p / index
    imports = (parameters["import_level"] + parameters["import_slope"] * (ratio - 1.0)
               - parameters["import_step"] * (1.0 - ndtr((p - parameters["import_step_price"]) / parameters["import_step_width"])))
    storage = parameters["storage_range"] * (2.0 * ndtr((ratio - parameters["storage_centre"]) / parameters["storage_width"]) - 1.0)
    return total + imports + storage + parameters["unmeasured"]


def _stopped(price, parameters: dict) -> np.ndarray:
    """GW of plant that is off at this price: all of it below zero, half of it at zero"""
    p = np.asarray(price, dtype=float)
    return parameters["stops_below_zero"] * ((p < -ZERO_BAND) + 0.5 * (np.abs(p) <= ZERO_BAND))


def supply_shift(demand, price, gas_price, carbon_price, neighbours=0.0, parameters: dict = PARAMETERS) -> np.ndarray:
    """
    How much more supply there was than the curves say, in GW, given the price the market cleared at.

    demand: residual load less the other plant, GW
    neighbours: the result of neighbour_imports, or 0 to leave the neighbours inside the shift
    """
    curve = supply(price, gas_price, carbon_price, parameters) - _stopped(price, parameters)
    return np.asarray(demand, dtype=float) - curve - np.asarray(neighbours, dtype=float)


def clearing_price(demand, gas_price, carbon_price, shift=0.0, neighbours=0.0, parameters: dict = PARAMETERS) -> np.ndarray:
    """
    The price in EUR/MWh at which supply plus shift meets demand. All arguments broadcast against each other.

    Where demand falls between what is supplied just above and just below zero, the price is zero: the plant that
    stops below zero makes a step in the supply there.
    """
    demand, gas, carbon, extra = np.broadcast_arrays(
        np.asarray(demand, dtype=float), np.asarray(gas_price, dtype=float), np.asarray(carbon_price, dtype=float),
        np.asarray(shift, dtype=float) + np.asarray(neighbours, dtype=float))

    def excess(price, stopped):
        return supply(price, gas, carbon, parameters) + extra - stopped - demand

    above = excess(np.full(demand.shape, ZERO_BAND), 0.0) < 0
    below = excess(np.full(demand.shape, -ZERO_BAND), parameters["stops_below_zero"]) > 0
    low = np.where(above, ZERO_BAND, -3000.0)
    high = np.where(above, 20000.0, -ZERO_BAND)
    stopped = np.where(below, parameters["stops_below_zero"], 0.0)
    for _ in range(60):
        middle = 0.5 * (low + high)
        short = excess(middle, stopped) < 0
        low = np.where(short, middle, low)
        high = np.where(short, high, middle)
    return np.where(above | below, 0.5 * (low + high), 0.0)

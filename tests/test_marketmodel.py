"""Tests for the market model: the supply curves and the clearing price."""

import numpy as np
import pytest

from predictor.model import marketmodel as mm

CARBON = 70.0
# the gas price at which the cost index is 90 EUR/MWh
GAS = (90.0 - 0.37 * CARBON) / 1.85


def test_cost_index():
    assert mm.cost_index(GAS, CARBON) == pytest.approx(90.0)


def test_supply_matches_the_fitted_curves():
    """Values worked out with the code the parameters were fitted with"""
    prices = np.array([-50.0, -5.0, 0.05, 20.0, 60.0, 90.0, 130.0, 250.0])
    expected = [-19.9994, -11.7961, -7.2591, -0.935, 10.4308, 24.2451, 43.0364, 71.2849]
    assert mm.supply(prices, GAS, CARBON) == pytest.approx(expected, abs=2e-3)


def test_clearing_price_matches_the_fitted_model():
    demand = np.array([-5.0, 5.0, 20.0, 35.0, 50.0])
    shift = np.array([0.0, 0.0, 2.0, -3.0, 0.0])
    expected = [2.721, 44.133, 77.375, 118.088, 149.743]
    assert mm.clearing_price(demand, GAS, CARBON, shift) == pytest.approx(expected, abs=5e-3)


def test_supply_rises_with_the_price():
    prices = np.linspace(-200.0, 600.0, 801)
    assert (np.diff(mm.supply(prices, GAS, CARBON)) > 0).all()


@pytest.mark.parametrize("price", [-80.0, -12.5, -1.0, 3.0, 45.0, 88.0, 140.0, 400.0])
def test_shift_and_clearing_are_inverse(price):
    """The shift read off a known price gives that price back"""
    demand, neighbours = 31.0, -1.5
    shift = mm.supply_shift(demand, price, GAS, CARBON, neighbours)
    assert mm.clearing_price(demand, GAS, CARBON, shift, neighbours) == pytest.approx(price, abs=1e-6)


def test_more_demand_or_less_supply_raises_the_price():
    demand = np.linspace(-15.0, 70.0, 60)
    assert (np.diff(mm.clearing_price(demand, GAS, CARBON)) >= 0).all()
    shift = np.linspace(-20.0, 20.0, 60)
    assert (np.diff(mm.clearing_price(30.0, GAS, CARBON, shift)) <= 0).all()


def test_price_is_zero_on_the_step_below_zero():
    """Between the supply just above zero and the supply just below it, less the plant that stops there"""
    just_above = float(mm.supply(mm.ZERO_BAND, GAS, CARBON))
    just_below = float(mm.supply(-mm.ZERO_BAND, GAS, CARBON)) - mm.PARAMETERS["stops_below_zero"]
    assert just_below < just_above
    middle = 0.5 * (just_above + just_below)
    assert mm.clearing_price(middle, GAS, CARBON) == 0.0
    assert mm.clearing_price(just_above + 0.5, GAS, CARBON) > mm.ZERO_BAND
    assert mm.clearing_price(just_below - 0.5, GAS, CARBON) < -mm.ZERO_BAND


def test_a_dearer_fuel_raises_the_price():
    assert mm.clearing_price(40.0, GAS + 10.0, CARBON) > mm.clearing_price(40.0, GAS, CARBON)
    assert mm.clearing_price(40.0, GAS, CARBON + 20.0) > mm.clearing_price(40.0, GAS, CARBON)


def test_neighbour_imports():
    assert mm.neighbour_imports(mm.NEIGHBOUR_MEAN) == pytest.approx(0.0)
    # a neighbour short of power leaves less for Germany
    belgium_short = mm.NEIGHBOUR_MEAN.copy()
    belgium_short[mm.NEIGHBOURS.index("BE")] += 2.0
    assert mm.neighbour_imports(belgium_short) < 0
    many = np.tile(mm.NEIGHBOUR_MEAN, (96, 1))
    assert mm.neighbour_imports(many).shape == (96,)


def test_arguments_broadcast():
    demand = np.full((3, 96), 30.0)
    gas = np.array([[25.0], [35.0], [45.0]])
    prices = mm.clearing_price(demand, gas, CARBON, shift=np.linspace(-2.0, 2.0, 96))
    assert prices.shape == (3, 96)
    assert (prices[0] < prices[2]).all()

"""Write the working-tree state for commit N (2, 3 or 4) of the PR, derived from the final file versions.
2: gas-relative model + 540 days, 3: + Europe-wide weather, 4: + North Sea locations (= final)."""
import os
import sys

final_dir, repo, n = sys.argv[1], sys.argv[2], int(sys.argv[3])


def final(path):
    return open(f"{final_dir}/{path}").read()


def cut(s, start, end):
    a = s.index(start)
    b = s.index(end, a)
    return s[:a] + s[b:]


def drop(s, old):
    assert old in s, old[:70]
    return s.replace(old, "")


def swap(s, old, new):
    assert old in s, old[:70]
    return s.replace(old, new)


out = {}

# ---------------------------------------------------------------- pricepredictor.py
s = final("predictor/model/pricepredictor.py")
if n < 3:
    s = drop(s, "from .europeweather import EuropeWeather\n")
    s = drop(s, '''
    # large-scale weather features, shared by the models of all regions. Set by whoever creates the models
    europe_weather: EuropeWeather | None = None
''')
    s = drop(s, "        self.europe_weather = other.europe_weather\n")
    s = drop(s, '''        if self.europe_weather is not None:
            europe_weather = await self.europe_weather.get_data(start, end)
            if europe_weather is not None:
                df = pd.concat([df, europe_weather], axis=1, sort=True)

''')
out["predictor/model/pricepredictor.py"] = s

# ---------------------------------------------------------------- priceapi.py / performance_testing.py
s = final("predictor/api/priceapi.py")
if n < 3:
    s = drop(s, "from predictor.model.europeweather import EuropeWeather\n")
    s = cut(s, "        # Large-scale weather features, built from the weather of all base regions.", "        # Stage-2 models are wired to ALL available base models")
out["predictor/api/priceapi.py"] = s

s = final("predictor/performance_testing.py")
if n < 3:
    s = drop(s, "from model.europeweather import EuropeWeather\n")
    s = cut(s, "\n    # Large-scale weather features, built from the weather of all regions", "\n    # Stage 2: only for eval regions")
out["predictor/performance_testing.py"] = s

# ---------------------------------------------------------------- weatherstore.py
if n >= 3:
    s = final("predictor/model/weatherstore.py")
    if n < 4:
        s = swap(s, "from typing import Self, override\n", "from typing import override\n")
        s = cut(s, "    def expected_columns(self)", "    async def refresh_range(")
        s = swap(s, "self.region.weather_latitudes", "self.region.latitudes")
        s = swap(s, "self.region.weather_longitudes", "self.region.longitudes")
    out["predictor/model/weatherstore.py"] = s

# ---------------------------------------------------------------- tests
s = final("tests/test_pricepredictor.py")
if n < 3:
    s = drop(s, "from predictor.model.europeweather import EuropeWeather\n")
    s = cut(s, "    @pytest.mark.asyncio\n    async def test_dataframe_with_europe_weather(", "    @pytest.mark.asyncio\n    async def test_dataframe_without_other_regions(")
    s = swap(s, '''c.startswith(("eu_", "cross_price_"))''', '''c.startswith("cross_price_")''')
    s = cut(s, "    def test_use_datastores_from_shares_europe_weather(", "class TestPricePredictorCleanup:")
out["tests/test_pricepredictor.py"] = s
if n >= 3:
    out["tests/test_europeweather.py"] = final("tests/test_europeweather.py")
    out["predictor/model/europeweather.py"] = final("predictor/model/europeweather.py")
if n >= 4:
    out["tests/test_weatherstore.py"] = final("tests/test_weatherstore.py")
    out["predictor/model/priceregion.py"] = final("predictor/model/priceregion.py")

# ---------------------------------------------------------------- README.md
s = final("README.md")
if n < 4:
    s = swap(s, "This serves as the main data source. For Germany, a few additional locations in the North Sea are sampled, too.\n",
             "This serves as the main data source.\n")
if n < 3:
    s = cut(s, "- Europe-wide weather: wind power potential", "\nOutput:")
    s = swap(s, "(180-day training window, no gas-relative model, no Europe-wide weather features)",
             "(180-day training window, no gas-relative model)")
out["README.md"] = s

for path, content in out.items():
    open(f"{repo}/{path}", "w").write(content)
print("wrote", len(out), "files for commit", n)

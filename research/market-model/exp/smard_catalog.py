import sys, json, collections
d = sys.argv[1]; cfg = json.load(open(f"{d}/market_data_configuration.json")); pp = json.load(open(f"{d}/power_plant_data_power_plant_metadata.json"))
strip = lambda s: str(s).split(".", 1)[-1]; cat = []
for main in cfg["main"]:
    print(f"\n## {strip(main['name'])}")
    for sub in main.get("sub", []):
        mods = []
        for grp, lst in (sub.get("module") or {}).items():
            for m in (lst if isinstance(lst, list) else [lst]):
                if isinstance(m, dict): mods.append(m)
        print(f"  - {strip(sub['name'])}: {len(mods)} series")
        for m in mods:
            cat.append({"main": strip(main["name"]), "sub": strip(sub["name"]), "id": m.get("id"), "data_id": m.get("data_id"), "name": strip(m.get("name", "")), "region": m.get("region"), "unit": m.get("unit"), "resolution": m.get("resolution")})
        names = [f"{strip(m.get('name', ''))} [{m.get('data_id')}]" for m in mods]; print("      " + "; ".join(names)[:1500])
json.dump(cat, open(f"{d}/catalog.json", "w"), ensure_ascii=False, indent=0); print("\nexample module entry:", json.dumps([m for m in cfg["main"][0]["sub"][0]["module"]["other"]][0], ensure_ascii=False)[:500])
# power plants
blocks = [dict(b, plant=strip(p["name"]), company=p.get("company"), region=p.get("regionId"), city=strip(p.get("city", ""))) for p in pp["plants"] for b in p.get("blocks", [])]
print(f"\n## power plants: {len(pp['plants'])} plants, {len(blocks)} blocks; keys of a block: {list(blocks[0].keys())}")
agg = collections.defaultdict(lambda: [0, 0.0])
for b in blocks:
    k = (strip(b["resource"]), strip(b["status"]), strip(b.get("chp", ""))); agg[k][0] += 1; agg[k][1] += float(b.get("power") or 0)
for k in sorted(agg, key=lambda k: -agg[k][1])[:40]: print(f"   {k[0]:28s} {k[1]:36s} CHP {k[2]:5s}: {agg[k][0]:3d} blocks, {agg[k][1] / 1000:6.1f} GW")
json.dump(blocks, open(f"{d}/blocks.json", "w"), ensure_ascii=False)

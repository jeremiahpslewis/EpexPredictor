"""Publication-timing probe: which ENTSO-E day-ahead documents (load A65/A01, wind+solar A69/A01) exist now for
D = tomorrow and D+1, per zone, with the document createdDateTime and the last valid time. Key read from SP/entsoe/key,
never printed. One line per (zone, doc, day) appended to SP/team/I/probe/entsoe_probe.log. usage: entsoe_probe.py SP"""
import sys, re, datetime as dt, requests
SP = sys.argv[1]
KEY = open(f"{SP}/entsoe/key").read().strip()
EIC = {"DE_LU": "10Y1001A1001A82H", "FR": "10YFR-RTE------C", "NL": "10YNL----------L", "BE": "10YBE----------2"}
EIC_MORE = {
       "AT": "10YAT-APG------L", "PL": "10YPL-AREA-----S", "CZ": "10YCZ-CEPS-----N", "DK_1": "10YDK-1--------W", "CH": "10YCH-SWISSGRIDZ"}
now = dt.datetime.now(dt.timezone.utc)
out = open(f"{SP}/team/I/probe/entsoe_probe.log", "a")
# NB createdDateTime in API answers is usually the answer time, not the publication time: the useful signal is
# whether D's document EXISTS at probe time (run before the gate, D = tomorrow).
for k in (1, 2):
    d = (now + dt.timedelta(days=k)).date()
    s = dt.datetime(d.year, d.month, d.day) - dt.timedelta(hours=1); e = s + dt.timedelta(days=1)
    for z, eic in EIC.items():
        for doc, par in (("A65", {"documentType": "A65", "processType": "A01", "outBiddingZone_Domain": eic}),
                         ("A69", {"documentType": "A69", "processType": "A01", "in_Domain": eic})):
            p = dict(par, securityToken=KEY, periodStart=s.strftime("%Y%m%d%H%M"), periodEnd=e.strftime("%Y%m%d%H%M"))
            try:
                r = requests.get("https://web-api.tp.entsoe.eu/api", params=p, timeout=40)
                t = r.text
                created = re.findall(r"<createdDateTime>([^<]+)<", t)
                ends = re.findall(r"<end>([^<]+)</end>", t)
                npts = t.count("<Point>")
                msg = f"status {r.status_code} created {created[:1]} last_end {max(ends) if ends else None} points {npts}"
                if "Acknowledgement" in t:
                    msg += " ACK " + (re.findall(r"<text>([^<]+)<", t) or [""])[0][:80]
            except Exception as ex:
                msg = f"error {type(ex).__name__}"
            out.write(f"{now:%Y-%m-%d %H:%M} UTC | {z} {doc} D={d} | {msg}\n"); out.flush()
out.close()

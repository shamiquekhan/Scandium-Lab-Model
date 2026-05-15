from mp_api.client import MPRester
import json

API_KEY = "hvdrSrYdq5WAmY1Jkifxy6cTM97JVxYi"
with MPRester(api_key=API_KEY) as mpr:
    results = mpr.materials.search(material_ids=["mp-23907"])
    if results:
        s = results[0].structure
        with open("mp-23907-pymatgen.json","w",encoding="utf-8") as f:
            json.dump(s.as_dict(), f, indent=2)
        print("Wrote mp-23907-pymatgen.json")
    else:
        print("Material mp-23907 not found")

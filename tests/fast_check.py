from pathlib import Path

sp = Path("/home/jha/jha/python_scripts/CRYSTALdataGen/structures/AlN/singlepoints_mixed150")
outs = sorted(sp.rglob("*sp.out"))
print("out files found:", len(outs))
for p in outs[:10]:
    print(p.name)

"""
Experiment T (NOTEBOOK entry 45): artefact tolerance outside LIRF of 15 or 30 s (argument), LIRF stays 60 s. Writes
exp_t15.py or exp_t30.py by patching taxiout.py. Written for taxiout.py as of 9 Oct 2026 (A+R, md5
1f6313c7f7a3fa107572f86da458ad37). Its edits also apply to the final taxiout.py; if the text differs it stops
without writing anything.

    python mk_exp_t.py 15 && python exp_t15.py validate --aobt
"""
import sys

tol = int(sys.argv[1])
s = open("taxiout.py").read()
edits = [
('''    """1 when the reported taxi equals the scheduled delay (+-60 s at LIRF, +-1 s elsewhere)"""''',
 f'''    """1 when the reported taxi equals the scheduled delay (+-60 s at LIRF, +-{tol} s elsewhere)"""'''),
('''    tol = np.where(df["ADEP_mvt"].astype(str).to_numpy() == "LIRF", 60.0, 1.0)''',
 f'''    tol = np.where(df["ADEP_mvt"].astype(str).to_numpy() == "LIRF", 60.0, {tol:.1f})'''),
('return f"taxiout_s{seeds}"', f'return f"expt{tol}_s{{seeds}}"'),
]
for old, new in edits:
    if s.count(old) != 1:
        raise SystemExit(f"taxiout.py is not as expected, nothing written. Problem near: {old[:60]!r}")
    s = s.replace(old, new)
open(f"exp_t{tol}.py", "w").write(s)
print(f"wrote exp_t{tol}.py")

"""Serial GPU queue: two epochs each first, then finish both and test once."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from soccer_gif_reid.common import atomic_json

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/"reports/arg_fra_validation_v3"


def main():
    os.chdir(ROOT)
    cfg="configs/arg_fra_validation_v3.yaml"
    phases=[("clip",2),("ap3d",2),("clip",None),("ap3d",None)]
    status={"supervisor_pid":os.getpid(),"started_at":datetime.now(timezone.utc).isoformat(),
            "config":cfg,"phases":[{"model":k,"until_epoch":e} for k,e in phases],"completed_phases":[]}
    try:
        for index,(kind,until) in enumerate(phases):
            run=ROOT/"runs/arg_fra_validation_v3"/kind
            previous=json.loads((run/"status.json").read_text()) if (run/"status.json").exists() else {}
            if previous.get("status")=="complete" or (until and previous.get("completed_epochs",0)>=until):
                status["completed_phases"].append(index);continue
            command=[sys.executable,"-m","experiments.arg_fra_validation_v3.train","--config",cfg,"--kind",kind]
            if until:command += ["--until-epoch",str(until)]
            if (run/"last.pt").exists():command.append("--resume")
            status.update(status="training",current_phase=index,kind=kind,command=command)
            atomic_json(REPORT/"queue_status.json",status)
            with (REPORT/f"{kind}_training.log").open("a") as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                status["child_pid"]=child.pid;atomic_json(REPORT/"queue_status.json",status)
                code=child.wait()
            if code:raise RuntimeError(f"{kind} phase {index} failed, exit {code}; see {kind}_training.log")
            status["completed_phases"].append(index)
        for kind in ("clip","ap3d"):
            target=REPORT/f"{kind}_test_metrics.json"
            if target.exists():continue
            status.update(status="final_test",kind=kind)
            atomic_json(REPORT/"queue_status.json",status)
            command=[sys.executable,"-m","experiments.ap3d_no_quality_v2.evaluate","--checkpoint",
                     f"runs/arg_fra_validation_v3/{kind}/best.pt","--split","test","--output",str(target)]
            with (REPORT/f"{kind}_test.log").open("a") as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        status.update(status="complete",finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        status.update(status="failed",error=str(exc),finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(REPORT/"queue_status.json",status)
        raise
    atomic_json(REPORT/"queue_status.json",status)


if __name__=="__main__":main()

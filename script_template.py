import gzip
import base64
import os
from pathlib import Path
from typing import Dict


# this is base64 encoded source code
file_data: Dict = {file_data}


for path, encoded in file_data.items():
    print(path)
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(gzip.decompress(base64.b64decode(encoded)))


def run(command):
    os.system('export PYTHONPATH=${PYTHONPATH}:/kaggle/working && ' + command)


run('python setup.py develop --install-dir /kaggle/working')
run('python -m imet.n05_main --checkpoint "../input/imetweightn01z3/eff_b3.pth" --model eff_b3 --scale-size 600 --crop-size 300')
run('python -m imet.n05_main --checkpoint "../input/imetweightn01z3/dn161.pth" --model dn161 --scale-size 576 --crop-size 288')
run('python -m imet.n05_main --checkpoint "../input/imetweightn01z3/se_resnext50_288.pth" --model se_resnext50 --scale-size 576 --crop-size 288')
run('python -m imet.n05_main --checkpoint "../input/imetweightn01z3/se_resnext50.pth" --model se_resnext50')
run('python -m imet.n05_main --checkpoint "../input/imetweightn01z3/se_resnext101.pth" --model se_resnext101')
run('python -m imet.n06_make_submission')

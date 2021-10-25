import anabel.damage

DmgData = {
    "dFun": "MBeta",
    "dp": [[[1, 1], [1, 1]], 
           [[1, 1], [1, 1]]],
    "Cd0": [[0, 0], [0, 0]],
    "Cd1": [[35, 35], [35, 35]],
    "Ccd": [[1, 1], [1, 1]],
    "Cwc": [[0.3, 0.3], [0.3, 0.3]],
    "n": 2,
    "DOpt": "total",
    "Frac": [
        [{"Activ": False, "psiF": 35, "psiU": 35}]*2,
        [{"Activ": False, "psiF": 35, "psiU": 35}]*2,
    ],
    "gtol": 1e-8,
    "dtol": 0.0001,
    "Cin_pp": [[ 0,  0], [ 0,  0]],
    "Cin_pn": [[ 0,  0], [ 0,  0]],
    "Cin_np": [[ 0,  0], [ 0,  0]],
    "Cin_nn": [[ 0,  0], [ 0,  0]],
    "psiF":   [[35, 35], [35, 35]],
    "psiU":   [[35, 35], [35, 35]],
    "psi_d0": [[ 0,  0], [ 0,  0]],
    "psi_d1": [[350, 350], [70, 70]],
}

def test_section():
    f = anabel.damage.DmgEvowNpnd(**DmgData)


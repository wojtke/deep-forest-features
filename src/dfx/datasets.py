"""The 32 datasets of plan Table 2, plus tiny smoke datasets for tests.

Each `Spec` names its raw archive (URL + pinned sha256), a reader returning one frame with every
column, the target, and the columns excluded by plan §4 (ids, timestamps, sampling weights,
columns that reveal the target). See docs/data-audit.md for what each spec produces.
"""
import io
import re
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Dataset, Spec, related_groups

UCI = "https://archive.ics.uci.edu/static/public"


def zip_bytes(raw: Path, member: str, inner: str | None = None) -> bytes:
    """One member of the downloaded archive raw/<name>.zip, or of a zip `inner` inside it."""
    with zipfile.ZipFile(raw / f"{raw.name}.zip") as z:
        if inner:
            z = zipfile.ZipFile(io.BytesIO(z.read(inner)))
        return z.read(next(n for n in z.namelist() if n.endswith(member)))


def from_zip(raw: Path, member: str, inner: str | None = None, **read) -> pd.DataFrame:
    """Read a CSV-like or Excel member (Excel: first sheet unless `sheet_name` is given)."""
    data = io.BytesIO(zip_bytes(raw, member, inner))
    return pd.read_excel(data, **read) if member.endswith((".xls", ".xlsx")) else pd.read_csv(data, **read)


# --- classification -------------------------------------------------------------------------

def read_sonar(raw: Path) -> pd.DataFrame:
    frame = from_zip(raw, "sonar.all-data", header=None)
    return frame.rename(columns={60: "class"}).rename(columns=str)


def read_spambase(raw: Path) -> pd.DataFrame:
    doc = zip_bytes(raw, "spambase.names").decode().splitlines()
    names = [line.split(":")[0] for line in doc if line.rstrip().endswith("continuous.")]
    return from_zip(raw, "spambase.data", header=None, names=[*names, "class"])


def read_qsar_biodeg(raw: Path) -> pd.DataFrame:
    names = ["SpMax_L", "J_Dz(e)", "nHM", "F01[N-N]", "F04[C-N]", "NssssC", "nCb-", "C%", "nCp", "nO",
             "F03[C-N]", "SdssC", "HyWi_B(m)", "LOC", "SM6_L", "F03[C-O]", "Me", "Mi", "nN-N", "nArNO2",
             "nCRX3", "SpPosA_B(p)", "nCIR", "B01[C-Br]", "B03[C-Cl]", "N-073", "SpMax_A", "Psi_i_1d",
             "B04[C-Br]", "SdO", "TI2_L", "nCrt", "C-026", "F02[C-N]", "nHDon", "SpMax_B(m)", "Psi_i_A",
             "nN", "SM6_B(m)", "nArCOOR", "nX", "class"]
    return from_zip(raw, "biodeg.csv", sep=";", header=None, names=names)


def read_steel_plates(raw: Path) -> pd.DataFrame:
    names = zip_bytes(raw, "Faults27x7_var").decode().split()  # 27 features, then 7 faults
    frame = from_zip(raw, "Faults.NNA", sep=r"\s+", header=None, names=names)
    faults = frame[names[27:]]
    if not (faults.sum(axis=1) == 1).all():
        raise ValueError("steel_plates: a plate without exactly one fault")
    return frame[names[:27]].assign(fault=faults.idxmax(axis=1))


def read_vehicle(raw: Path) -> pd.DataFrame:
    names = ["COMPACTNESS", "CIRCULARITY", "DISTANCE CIRCULARITY", "RADIUS RATIO", "PR.AXIS ASPECT RATIO",
             "MAX.LENGTH ASPECT RATIO", "SCATTER RATIO", "ELONGATEDNESS", "PR.AXIS RECTANGULARITY",
             "MAX.LENGTH RECTANGULARITY", "SCALED VARIANCE ALONG MAJOR AXIS",
             "SCALED VARIANCE ALONG MINOR AXIS", "SCALED RADIUS OF GYRATION", "SKEWNESS ABOUT MAJOR AXIS",
             "SKEWNESS ABOUT MINOR AXIS", "KURTOSIS ABOUT MINOR AXIS", "KURTOSIS ABOUT MAJOR AXIS",
             "HOLLOWS RATIO", "class"]
    parts = [from_zip(raw, f"xa{c}.dat", sep=r"\s+", header=None, names=names) for c in "abcdefghi"]
    return pd.concat(parts, ignore_index=True)


def read_heart_statlog(raw: Path) -> pd.DataFrame:
    names = ["age", "sex", "chest-pain", "rest-bp", "serum-chol", "fasting-blood-sugar",
             "electrocardiographic", "max-heart-rate", "angina", "oldpeak", "slope", "major-vessels", "thal",
             "heart-disease"]
    return from_zip(raw, "heart.dat", sep=r"\s+", header=None, names=names)


def read_tictactoe(raw: Path) -> pd.DataFrame:
    names = [f"{row}-{col}-square" for row in ("top", "middle", "bottom") for col in ("left", "middle", "right")]
    return from_zip(raw, "tic-tac-toe.data", header=None, names=[*names, "class"])


def read_page_blocks(raw: Path) -> pd.DataFrame:
    names = ["height", "length", "area", "eccen", "p_black", "p_and", "mean_tr", "blackpix", "blackand",
             "wb_trans", "class"]
    packed = zip_bytes(raw, "page-blocks.data.Z")  # Unix compress: gzip can read it
    text = subprocess.run(["gzip", "-dc"], input=packed, capture_output=True, check=True).stdout
    return pd.read_csv(io.BytesIO(text), sep=r"\s+", header=None, names=names)


def read_german_credit(raw: Path) -> pd.DataFrame:
    names = ["checking_account", "duration", "credit_history", "purpose", "credit_amount", "savings",
             "employment_since", "installment_rate", "personal_status_sex", "other_debtors",
             "residence_since", "property", "age", "other_installment_plans", "housing", "existing_credits",
             "job", "people_liable", "telephone", "foreign_worker", "class"]  # class 1 good, 2 bad
    return from_zip(raw, "german.data", sep=" ", header=None, names=names)


def read_cardiotocography(raw: Path) -> pd.DataFrame:
    book = zip_bytes(raw, "CTG.xls")
    sheet = pd.read_excel(io.BytesIO(book), sheet_name="Data", header=1)
    # Columns 2-8 count AC ... DR per segment; the 21 UCI features (numbered 1-21 above the header)
    # repeat AC ... DP per second as "AC.1" ... "DP.1" and leave out DR. Unnamed columns are blank.
    frame = sheet.drop(columns=sheet.columns[2:9]).dropna(subset=["NSP"])  # sums below the data
    frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed")]
    frame = frame.rename(columns=lambda c: c.removesuffix(".1")).astype({"NSP": int}).reset_index(drop=True)
    # Segments of one recording (one exam) are related rows; the recording is named in "Raw Data".
    recordings = pd.read_excel(io.BytesIO(book), sheet_name="Raw Data").dropna(subset=["FileName"])
    if not all((recordings[c].to_numpy() == frame[c].to_numpy()).all() for c in ("b", "e", "LB", "NSP")):
        raise ValueError("cardiotocography: sheets Data and Raw Data rows differ")
    return frame.assign(recording=recordings["FileName"].to_numpy())


def read_phishing(raw: Path) -> pd.DataFrame:
    head, body = zip_bytes(raw, "Training Dataset.arff").decode().split("@data")
    return pd.read_csv(io.StringIO(body), header=None, names=re.findall(r"@attribute\s+(\S+)", head))


def read_spectf(raw: Path) -> pd.DataFrame:
    names = ["diagnosis", *(f"F{i}{s}" for i in range(1, 23) for s in "RS")]
    return pd.concat([from_zip(raw, part, header=None, names=names, skipinitialspace=True)
                      for part in ("SPECTF.train", "SPECTF.test")], ignore_index=True)


def read_adult(raw: Path) -> pd.DataFrame:
    names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
             "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
             "native-country", "income"]
    read = dict(header=None, names=names, skipinitialspace=True, na_values="?")
    frame = pd.concat([from_zip(raw, "adult.data", **read),
                       from_zip(raw, "adult.test", skiprows=1, **read)],  # line 1: "|1x3 Cross validator"
                      ignore_index=True)
    return frame.assign(income=frame["income"].str.rstrip("."))  # the test file writes "<=50K."


# --- regression -----------------------------------------------------------------------------

def read_abalone(raw: Path) -> pd.DataFrame:
    names = ["sex", "length", "diameter", "height", "whole_weight", "shucked_weight",
             "viscera_weight", "shell_weight", "rings"]
    return from_zip(raw, "abalone.data", header=None, names=names)


def read_auto_mpg(raw: Path) -> pd.DataFrame:
    names = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year",
             "origin", "car_name"]
    return from_zip(raw, "auto-mpg.data", sep=r"\s+", header=None, names=names, na_values="?")


def read_automobile(raw: Path) -> pd.DataFrame:
    names = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors",
             "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", "height",
             "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke",
             "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]
    return from_zip(raw, "imports-85.data", header=None, names=names, na_values="?")


def read_communities(raw: Path) -> pd.DataFrame:
    names = re.findall(r"@attribute (\S+)", zip_bytes(raw, "communities.names").decode())
    return from_zip(raw, "communities.data", header=None, names=names, na_values="?")


def read_servo(raw: Path) -> pd.DataFrame:
    names = ["motor", "screw", "pgain", "vgain", "class"]
    return from_zip(raw, "servo.data", header=None, names=names, skipinitialspace=True)


def read_superconductivity(raw: Path) -> pd.DataFrame:
    frame = from_zip(raw, "train.csv")
    formulas = from_zip(raw, "unique_m.csv", usecols=["critical_temp", "material"])
    if not frame["critical_temp"].equals(formulas["critical_temp"]):  # same rows in the same order
        raise ValueError("superconductivity: train.csv and unique_m.csv rows differ")
    return frame.assign(material=formulas["material"])


def read_air_quality(raw: Path) -> pd.DataFrame:
    frame = from_zip(raw, "AirQualityUCI.csv", sep=";", decimal=",")
    frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="all")  # empty ";;" columns and rows
    return frame.replace(-200, np.nan)  # -200 marks a missing value


def read_california(raw: Path) -> pd.DataFrame:
    from sklearn.datasets import fetch_california_housing

    bunch = fetch_california_housing(data_home=raw, as_frame=True)  # sklearn checks the archive hash
    return bunch.frame


SPECS = [
    # --- classification
    Spec("sonar", "classification", "class", read_sonar,
         url=f"{UCI}/151/connectionist+bench+sonar+mines+vs+rocks.zip",
         sha256="088b0b6813fb5ab84736bc2a1e3d6bb886e317520d2f2c04152e2ed65af7a18d",
         plan=(208, 60)),
    Spec("spambase", "classification", "class", read_spambase,
         url=f"{UCI}/94/spambase.zip",
         sha256="813ac1df8effac70463c09c9c4b11e8803eefcab54771af66150852bcdcd1636", plan=(4601, 57),
         note="class 1 = spam; names from spambase.names"),
    Spec("qsar_biodeg", "classification", "class", read_qsar_biodeg,
         url=f"{UCI}/254/qsar+biodegradation.zip",
         sha256="b5f7cf1b05564d94d65265a9a4aeae5fffcd15b8d3bcafe4225a8827d7ae08ac", plan=(1055, 41),
         note="names from the UCI variable information"),
    Spec("steel_plates", "classification", "fault", read_steel_plates,
         url=f"{UCI}/198/steel+plates+faults.zip",
         sha256="cb8eb9859198b63f053e443513036b401746fa517ef58bd17c846c6741c93919", plan=(1941, 27),
         note="target = which of the 7 one-hot fault columns is set (exactly one per row, checked); "
              "the 7 are not features"),
    Spec("vehicle", "classification", "class", read_vehicle,
         url=f"{UCI}/149/statlog+vehicle+silhouettes.zip",
         sha256="66ca807d5be5e3403cf035a579e85113880c3019d6cde825670deff05a4a71e0", plan=(846, 18),
         note="xaa.dat ... xai.dat concatenated"),
    Spec("heart_statlog", "classification", "heart-disease", read_heart_statlog,
         url=f"{UCI}/145/statlog+heart.zip",
         sha256="04e44018af4f4b32008029fbfb1079e112d3b372bb032dd1ae361c74d7528b2c", plan=(270, 13),
         categorical=("chest-pain", "electrocardiographic", "thal"),
         note="class 1 absence, 2 presence; chest-pain, electrocardiographic, thal nominal (heart.doc, UCI)"),
    Spec("tictactoe", "classification", "class", read_tictactoe,
         url=f"{UCI}/101/tic+tac+toe+endgame.zip",
         sha256="c615fcd01bc23ab99daa36bbc30d897d53f7d3e151257880cada888c5451a586", plan=(958, 9)),
    Spec("page_blocks", "classification", "class", read_page_blocks,
         url=f"{UCI}/78/page+blocks+classification.zip",
         sha256="b766faf029e05afd8c3969376fbdd01e377c7580eb88262772aa9c9e02fe28bf", plan=(5473, 10),
         note="page-blocks.data.Z decompressed with gzip -dc"),
    Spec("german_credit", "classification", "class", read_german_credit,
         url=f"{UCI}/144/statlog+german+credit+data.zip",
         sha256="e12d9d5def6845c0622634a1cd2ab87fa470668c4298f1ec52a4e403376a435b", plan=(1000, 20),
         note="german.data (symbolic codes A11 ...); class 1 good, 2 bad; names from german.doc"),
    Spec("cardiotocography", "classification", "NSP", read_cardiotocography,
         url=f"{UCI}/193/cardiotocography.zip",
         sha256="ba2853564e282779907242b7bd8a83265d9fc4e79bb008573967daa16677bb6c", plan=(2126, 21),
         drop=("b", "e", "CLASS", "A", "B", "C", "D", "E", "AD", "DE", "LD", "FS", "SUSP"), group="recording",
         note="PROPOSED (plan-silent, confirm): CLASS and the 10 pattern columns A ... SUSP (other expert "
              "labels of the same exam) dropped. CTG.xls sheet Data, UCI's 21 features; raw also counts b, e "
              "(segment start/end instants, excluded), those 11 labels and the recording; groups = the 352 "
              "recordings (sheet Raw Data, rows checked to align)"),
    Spec("climate_crashes", "classification", "outcome",
         lambda raw: from_zip(raw, "pop_failures.dat", sep=r"\s+"),
         url=f"{UCI}/252/climate+model+simulation+crashes.zip",
         sha256="fe8c0c2b79d1beedf31fbca2db8a97cec3c1b773ee29ed3357a2ff1dc8dd12d0", plan=(540, 18),
         drop=("Study", "Run"), note="raw also counts the Study and Run ids (plan does not); outcome 0 = crash"),
    Spec("phishing", "classification", "Result", read_phishing,
         url=f"{UCI}/327/phishing+websites.zip",
         sha256="514f582cb44cba17251baf1d59f64cddff16efa2b644d091ce8d2a1e8d492940", plan=(11055, 30),
         note="PROPOSED (plan-silent, confirm): the {-1, 0, 1} codes are numeric (ordinal), not categorical"),
    Spec("spectf", "classification", "diagnosis", read_spectf,
         url=f"{UCI}/96/spectf+heart.zip",
         sha256="24409e8bd112d372ec993304f029d61e5d2daa2de091399e2c3f84eed4f021ce", plan=(267, 44),
         note="SPECTF.train + SPECTF.test (80 + 187 rows)"),
    Spec("credit_default", "classification", "default payment next month",
         lambda raw: from_zip(raw, "default of credit card clients.xls", header=1),
         url=f"{UCI}/350/default+of+credit+card+clients.zip",
         sha256="56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602", plan=(30000, 23),
         drop=("ID",), note="raw also counts the ID column (plan does not); > 20,000 rows, subsampled"),
    Spec("bank_marketing", "classification", "y",
         lambda raw: from_zip(raw, "bank-full.csv", inner="bank.zip", sep=";"),
         url=f"{UCI}/222/bank+marketing.zip",
         sha256="e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4", plan=(45211, 16),
         drop=("duration",),
         note="PROPOSED (plan-silent, confirm): day and month of the last contact kept as features (calendar "
              "fields, not timestamps). bank-full.csv from the inner bank.zip; 'unknown' is a category; "
              "> 20,000 rows, subsampled"),
    Spec("adult", "classification", "income", read_adult,
         url=f"{UCI}/2/adult.zip",
         sha256="7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb", plan=(48842, 14),
         drop=("fnlwgt",), note="adult.data + adult.test, test labels without the trailing '.'; "
                                "> 20,000 rows, subsampled"),
    # --- regression
    Spec("abalone", "regression", "rings", read_abalone,
         url=f"{UCI}/1/abalone.zip",
         sha256="755a6a67c5b266961a3f149ea13be2cfb6e6c727e48cee3c83bc0b4526210ee4", plan=(4177, 8)),
    Spec("auto_mpg", "regression", "mpg", read_auto_mpg,
         url=f"{UCI}/9/auto+mpg.zip",
         sha256="5fc1507522b73b8050d3aa45df51eefe1b666ae5f428a5f1347b6483339f1003", plan=(398, 7),
         drop=("car_name",), note="raw also counts car_name (plan does not); 6 horsepower values missing"),
    Spec("automobile", "regression", "price", read_automobile,
         url=f"{UCI}/10/automobile.zip",
         sha256="0360d2379d2d505540f0c43bfe0ada96244ad2662faf9dada1629824948294ec", plan=(205, 25),
         note="4 rows without price dropped; num-of-doors, num-of-cylinders are words, so categorical"),
    Spec("communities", "regression", "ViolentCrimesPerPop", read_communities,
         url=f"{UCI}/183/communities+and+crime.zip",
         sha256="f0dcda420cdecc4dcf58939f6e6e637969d14ac8f95da1abde495088714c5038", plan=(1994, 127),
         drop=("state", "county", "community", "communityname", "fold"),
         note="the 5 non-predictive columns excluded; many police columns ~84% missing"),
    Spec("servo", "regression", "class", read_servo,
         url=f"{UCI}/87/servo.zip",
         sha256="02822bd9f00d8af7394163649e6398711da0a0b02170e380fd4d15b14e8a5659", plan=(167, 4)),
    Spec("energy", "regression", "Y1",
         lambda raw: from_zip(raw, "ENB2012_data.xlsx"),
         url=f"{UCI}/242/energy+efficiency.zip",
         sha256="499441eee27929a4b00417f58fd8c63c9cc14b8a71520cd0dd27fcb626738351", plan=(768, 8),
         drop=("Y2",), note="PROPOSED (plan-silent, confirm): Y2 (cooling load, a second target) dropped; "
                            "raw also counts it. First sheet (the others are empty)"),
    Spec("wine_red", "regression", "quality",
         lambda raw: from_zip(raw, "winequality-red.csv", sep=";"),
         url=f"{UCI}/186/wine+quality.zip",
         sha256="3ed56667f4b828242bd732d7d1dd7f2861e54432239d7fa63877014cbb0304d4", plan=(1599, 11),
         note="winequality-red.csv only"),
    Spec("concrete_slump", "regression", "Compressive Strength (28-day)(Mpa)",
         lambda raw: from_zip(raw, "slump_test.data").rename(columns=str.strip),
         url=f"{UCI}/182/concrete+slump+test.zip",
         sha256="68d6a6813261bf8ecf79d8d7cc82b665bd5e9e6800c7705a16e671688c78e491", plan=(103, 7),
         drop=("No", "SLUMP(cm)", "FLOW(cm)"),
         note="PROPOSED (plan-silent, confirm): SLUMP and FLOW (other targets) dropped; also the row number "
              "No (an id); raw also counts all three"),
    Spec("real_estate", "regression", "Y house price of unit area",
         lambda raw: from_zip(raw, "Real estate valuation data set.xlsx"),
         url=f"{UCI}/477/real+estate+valuation+data+set.zip",
         sha256="aa437bdac3ca23200258a0a58d251c0af90651aeaf7c7a07aaa714f0f9f25e2c", plan=(414, 6),
         drop=("No", "X1 transaction date"),
         note="PROPOSED (plan-silent, confirm): X1 transaction date (year + month) excluded as a timestamp. "
              "Raw also counts the row number No (plan does not), excluded as an id"),
    Spec("parkinsons", "regression", "total_UPDRS",
         lambda raw: from_zip(raw, "parkinsons_updrs.data"),
         url=f"{UCI}/189/parkinsons+telemonitoring.zip",
         sha256="2f82bb0ef96fa7d8d7edf4d97b89173e07c2cca7723440a64bc38918a83345df", plan=(5875, 19),
         drop=("motor_UPDRS", "test_time"), group="subject#",
         note="PROPOSED (plan-silent, confirm): test_time (days since recruitment) excluded as a timestamp. "
              "Raw also counts subject# and motor_UPDRS (plan does not); groups = the 42 patients"),
    Spec("student_math", "regression", "G3",
         lambda raw: from_zip(raw, "student-mat.csv", inner="student.zip", sep=";"),
         url=f"{UCI}/320/student+performance.zip",
         sha256="82ae9d66437b9808df42e8c89d2bb179c46e9cfbcf06f38abc1d20b3b747e177", plan=(395, 30),
         drop=("G1", "G2"), note="student-mat.csv from the inner student.zip; raw also counts G1, G2 (plan does not)"),
    Spec("superconductivity", "regression", "critical_temp", read_superconductivity,
         url=f"{UCI}/464/superconductivty+data.zip",
         sha256="87f4490d73390ff94ee01dbf0d7d32abc80b22f2c803d471765cfc46a9f6371e", plan=(21263, 81),
         group="material",
         note="raw also counts material, the formula from unique_m.csv (same row order, checked on "
              "critical_temp); groups = materials; > 20,000 rows, subsampled"),
    Spec("bike_sharing", "regression", "cnt",
         lambda raw: from_zip(raw, "hour.csv"),
         url=f"{UCI}/275/bike+sharing+dataset.zip",
         sha256="b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401", plan=(17379, 13),
         drop=("instant", "casual", "registered"), group="dteday",
         categorical=("season", "yr", "mnth", "hr", "weekday", "weathersit"),
         note="PROPOSED (plan-silent, confirm): calendar fields season, yr, mnth, hr, weekday kept as features "
              "(not timestamps). hour.csv; raw also counts instant, casual, registered (plan does not); "
              "groups = days; categorical as in the UCI table"),
    Spec("ccpp", "regression", "PE",
         lambda raw: from_zip(raw, "Folds5x2_pp.xlsx"),
         url=f"{UCI}/294/combined+cycle+power+plant.zip",
         sha256="cc7b2a4977c0a44e8221c91d9a7e5746b3c68186cff7e5c61c70af6432b98c7a", plan=(9568, 4),
         note="first sheet only; the other 4 hold the same rows reshuffled"),
    Spec("air_quality", "regression", "CO(GT)", read_air_quality,
         url=f"{UCI}/360/air+quality.zip",
         sha256="d4a64013fb385288a8a48d9d193ca7079b2e1bbddf6f8d458feb8c08ab2b8a2a", plan=(9357, 14),
         drop=("Time", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"), group="Date",
         note="PROPOSED (plan-silent, confirm): the other reference analyzers (GT columns, other targets "
              "like Energy Y2) dropped; Time (hour) excluded as a timestamp; the rows where every remaining "
              "sensor is missing dropped. -200 = missing; rows without CO(GT) dropped; groups = days"),
    Spec("california", "regression", "MedHouseVal", read_california, plan=(20640, 8),
         note="scikit-learn copy of the StatLib data (scikit-learn checks its hash); > 20,000 rows, subsampled"),
]


# --- smoke datasets: no download, built in memory ---------------------------------------------

def smoke_dataset(name: str) -> Dataset:
    rng = np.random.default_rng(0)
    if name == "smoke_xor":  # pure interaction: no marginal effect (plan §1 example)
        X = rng.normal(size=(300, 6))
        y = ((X[:, 0] * X[:, 1] > 0) ^ (rng.random(300) < 0.1)).astype(int)
        classes, task = ["neg", "pos"], "classification"
    elif name == "smoke_pair":
        X = rng.normal(size=(300, 6))
        y = X[:, 0] + 3 * X[:, 1] * X[:, 2] + rng.normal(scale=0.5, size=300)
        classes, task = [], "regression"
    elif name == "smoke_mixed":  # categorical column with missing values, three classes
        X = rng.normal(size=(300, 5))
        X[:, 0] = rng.integers(0, 3, 300)
        y = (X[:, 0] + (X[:, 1] > 0)).astype(int) % 3
        X[rng.random(X.shape) < 0.05] = np.nan
        classes, task = ["a", "b", "c"], "classification"
    elif name == "smoke_diabetes":
        from sklearn.datasets import load_diabetes

        X, y = load_diabetes(return_X_y=True)
        classes, task = [], "regression"
    else:
        raise KeyError(name)
    categorical = np.zeros(X.shape[1], dtype=bool)
    categorical[0] = name == "smoke_mixed"
    groups = related_groups(X)
    return Dataset(name, task, X, y, groups, np.arange(len(y)), [f"x{j}" for j in range(X.shape[1])],
                   categorical, classes, f"smoke-{name}")


SMOKE = {"smoke_xor", "smoke_pair", "smoke_mixed", "smoke_diabetes"}  # built in memory, for tests
DATASETS = {spec.name: spec for spec in SPECS}
ALL = list(DATASETS)  # what experiments mean by datasets = "all"

import os, sys
os.environ["PYSPARK_PYTHON"] = sys.executable          # avoids "Python worker failed to connect" on Windows
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from pyspark.sql import SparkSession

spark = (SparkSession.builder.master("local[*]").appName("smoke")
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         .config("spark.driver.memory", "4g").getOrCreate())

os.makedirs("data/smoke", exist_ok=True)
n = 200
vals = np.random.rand(n * 1250).astype(np.float32)
offs = np.arange(0, (n + 1) * 1250, 1250, dtype=np.int32)
pq.write_table(pa.table({"id": np.arange(n),
                         "ppg": pa.ListArray.from_arrays(pa.array(offs), pa.array(vals))}),
               "data/smoke/t.parquet")

df = spark.read.parquet("data/smoke")
df.printSchema()
print("rows:", df.count())

def f(it):
    for pdf in it:
        yield pd.DataFrame({"id": pdf["id"],
                            "mean": pdf["ppg"].apply(lambda a: float(np.mean(a)))})

print(df.mapInPandas(f, schema="id long, mean double").limit(3).toPandas())
spark.stop()
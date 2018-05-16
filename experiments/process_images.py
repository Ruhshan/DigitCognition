import cv2
import glob
import numpy as np
import pandas as pd
import asyncio
from timeit import default_timer

SHAPES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three":3,
    "four":4,
    "five":5,
    "six":6,
    "seven":7,
    "eight":8,
    "nine":9,


}

table=[]

async def read_and_process(img_path, label):
    img = cv2.imread(img_path, 0)
    #img = cv2.resize(img,(28,28))
    flat_img = img.flatten()
    row = np.append(flat_img, label)
    #print(row.shape)
    return row

# async def driver(target_files, label):
async def driver(shape):
    target_files = [f for f in glob.glob('data/train/{}/*'.format(shape))]
    label = SHAPES[shape]
    global table
    tasks = [read_and_process(img_path, label) for img_path in target_files]
    for row in await asyncio.gather(*tasks):
        table.append(row)

async def make_row(row):
    row=row.astype(str)
    row = ';'.join(row)+"\n"
    return row

async def table_writer(table):
    tasks = [make_row(row) for row in table]
    for row in await asyncio.gather(*tasks):
        o=open("data.csv",'a')
        o.write(row)
    o.close()



if __name__ == "__main__":

    start_time = default_timer()

    for shape, label in SHAPES.items():
        loop = asyncio.get_event_loop()
        loop.run_until_complete(driver(shape))
    end_time = default_timer()

    print("process time",end_time - start_time)
    start_time = default_timer()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(table_writer(table))
    end_time = default_timer()

    print("Writing time", end_time-start_time)
    df = pd.DataFrame(table)
    df.to_csv("data_async.csv", header=None)

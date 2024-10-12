#!/usr/bin/env python
import math
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from shutil import which


def count_sequences(file):
    n = 0
    fasta = open(file, "r")
    for line in fasta:
        if line.startswith(">"):
            n += 1
    fasta.close()
    return n


def run_prodigal(workDir, currentId, chunkFile):
    cmd = ["prodigal-gv", "-q", "-p", "meta", "-i", chunkFile.name]
    cmd.append("-a")
    cmd.append(workDir.name + "/chunk" + str(currentId) + ".faa")
    cmd.append("-f")
    cmd.append('gbk')
    cmd.append("-o")
    cmd.append(workDir.name + "/chunk" + str(currentId) + ".out")

    subprocess.run(cmd, shell=False, check=True)


def append_fasta_file(file, startNum, targetFile):
    pattern = re.compile(r"(.*ID=)(\d+)_(\d+)(.*)")
    with open(targetFile, "a") as trgt:
        with open(file, "r") as input:
            for line in input:
                if line.startswith(">"):
                    match = re.match(pattern, line)
                    if match and match.group(3) == "1":
                        startNum = startNum + 1
                    line = (
                        match.group(1)
                        + str(startNum)
                        + "_"
                        + match.group(3)
                        + match.group(4)
                    )
                trgt.write(line.strip() + "\n")



def main(queryFile, proteinFile, threads):


    if which("prodigal-gv") is None:
        raise ValueError("prodigal-gv not found in the PATH.")



    seqCnt = 0
    currentChunk = 1

    workDir = tempfile.TemporaryDirectory()
    executor = ThreadPoolExecutor(max_workers=threads)
    currentFile = open(workDir.name + "/chunk" + str(currentChunk), "w")


    n_sequences = count_sequences(queryFile)
    seqsPerChunk = math.ceil(n_sequences / threads)

    # Determine if the file is compressed and open it.
    fasta = open(queryFile, "r")

    for line in fasta:
        if line[0] == ">" and seqCnt == seqsPerChunk:
            currentFile.close()
            executor.submit(run_prodigal, workDir, currentChunk, currentFile)
            currentFile = None
            seqCnt = 0
            currentChunk += 1
        if currentFile is None:
            currentFile = open(workDir.name + "/chunk" + str(currentChunk), "w")
        currentFile.write(line)
        if line[0] == ">":
            seqCnt += 1

    fasta.close()

    if seqCnt > 0:
        currentFile.close()
        executor.submit(run_prodigal, workDir, currentChunk, currentFile)

    # await completion of tasks
    executor.shutdown(wait=True)

    # collect output
    protIdStart = 0
    for cur in range(1, currentChunk + 1):
        if proteinFile:
            append_fasta_file(
                workDir.name + "/chunk" + str(cur) + ".faa", protIdStart, proteinFile
            )
        protIdStart += seqsPerChunk


if __name__ == "__main__":
    main()

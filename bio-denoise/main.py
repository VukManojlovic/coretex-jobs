from typing import Optional
from pathlib import Path
from zipfile import ZipFile

import logging
import csv

from coretex import CustomDataset, CustomSample, Experiment, folder_manager
from coretex.project import initializeProject
from coretex.bioinformatics import ctx_qiime2


FORWARD_SUMMARY_NAME = "forward-seven-number-summaries.tsv"
REVERSE_SUMMARY_NAME = "reverse-seven-number-summaries.tsv"


def isPairedEnd(demuxSample: CustomSample):
    # In order to determine whether we are dealing with paired-end
    # sequences, this function unzips the qiime artifact and
    # reads the metadata, looking for the second (type) row, which will have
    # "PairedEnd" somewhere if it's paired-end

    demuxTemp = folder_manager.createTempFolder("demux")
    qzaPath = list(demuxSample.path.iterdir())[0]

    with ZipFile(qzaPath, "r") as qzaFile:
        qzaFile.extractall(demuxTemp)

    metadataPath = list(demuxTemp.rglob("*metadata.yaml"))[0]

    with metadataPath.open("r") as metadata:
        return "PairedEnd" in metadata.readlines()[1]


def dada2DenoiseSingleSample(
    sample: CustomSample,
    outputDir: Path,
    pairedEnd: bool,
    trimLeftF: int,
    truncLenF: int,
    trimLeftR: Optional[int],
    truncLenR: Optional[int]
) -> Path:

    samplePath = Path(sample.path)
    demuxPath = samplePath / "demux.qza"

    representativeSequencesPath = outputDir / "rep-seqs.qza"
    tablePath = outputDir / "table.qza"
    denoisingStatsPath = outputDir / "stats.qza"

    if pairedEnd:
        if trimLeftR is None or truncLenR is None:
            raise ValueError(f">> [Microbiome analysis] Required arguments for paired-end denoising trimLeftR and truncLenR must not be None. trimLeftR: \"{trimLeftR}\", truncLenR \"{truncLenR}\"")

        logging.info(">> [Microbiome analysis] Denoising paired-end sequences")
        ctx_qiime2.dada2DenoisePaired(
            str(demuxPath),
            trimLeftF,
            trimLeftR,
            truncLenF,
            truncLenR,
            str(representativeSequencesPath),
            str(tablePath),
            str(denoisingStatsPath)
        )
    else:
        logging.info(">> [Microbiome analysis] Denoising single-end sequences")
        ctx_qiime2.dada2DenoiseSingle(
            str(demuxPath),
            trimLeftF,
            truncLenF,
            str(representativeSequencesPath),
            str(tablePath),
            str(denoisingStatsPath)
        )

    denoiseOutput = outputDir / "denoise-output.zip"

    with ZipFile(denoiseOutput, "w") as denoiseFile:
        denoiseFile.write(representativeSequencesPath, "rep-seqs.qza")
        denoiseFile.write(tablePath, "table.qza")
        denoiseFile.write(denoisingStatsPath, "stats.qza")

    return denoiseOutput


def metadataTabulateSample(sample: CustomSample, outputDir: Path) -> Path:
    denoisingStatsPath = Path(sample.path) / "stats.qza"
    visualizationPath = outputDir / "stats.qzv"

    ctx_qiime2.metadataTabulate(str(denoisingStatsPath), str(visualizationPath))
    return visualizationPath


def featureTableSummarizeSample(sample: CustomSample, metadataPath: Path, outputDir: Path) -> Path:
    tablePath = Path(sample.path) / "table.qza"
    visualizationPath = outputDir / "table.qzv"

    ctx_qiime2.featureTableSummarize(str(tablePath), str(visualizationPath), str(metadataPath))
    return visualizationPath


def featureTableTabulateSeqsSample(sample: CustomSample, outputDir: Path) -> Path:
    inputPath = Path(sample.path) / "rep-seqs.qza"
    visualizationPath = outputDir / "rep-seqs.qzv"

    ctx_qiime2.featureTableTabulateSeqs(str(inputPath), str(visualizationPath))
    return visualizationPath


def determineTruncLen(sample: CustomSample, forward: bool) -> int:
    sample.unzip()

    summariesFileName = FORWARD_SUMMARY_NAME if forward else REVERSE_SUMMARY_NAME
    summariesTsv = list(sample.path.rglob(summariesFileName))[0]

    with summariesTsv.open("r") as file:
        summaries = list(csv.reader(file, delimiter = "\t"))

    # summaries will allways have the median quality at row 5 of the csv
    medianQualitiesStr = summaries[5]
    medianQualitiesStr.pop(0)  # The first value will be "50%" and not a quality score
    medianQualities = [float(x) for x in medianQualitiesStr]

    truncLen: Optional[int] = None
    highestScore = max(medianQualities)

    for index, qualityScore in enumerate(medianQualities):
        if qualityScore < highestScore * 0.7:
            truncLen = index
            break

    if not truncLen:
        raise RuntimeError(">> [Microbiome Analysis] Forward read truncLen could not be determined automatically")

    return truncLen


def processSample(
    index: int,
    sample: CustomSample,
    importedSample: CustomSample,
    summarySample: CustomSample,
    experiment: Experiment,
    outputDataset: CustomDataset,
    outputDir: Path
) -> None:

    sampleOutputDir = outputDir / str(sample.id)
    sampleOutputDir.mkdir()

    pairedEnd = isPairedEnd(sample)

    # First step:
    # Denoise the demultiplexed sample generated by the previous step in the pipeline
    trimLeftF = experiment.parameters["trimLeftF"]
    if trimLeftF is None:
        trimLeftF = 0

    trimLeftR = experiment.parameters["trimLeftR"]
    if trimLeftR is None:
        trimLeftR = 0

    # In case truncLen is not provided it will be determined automatically
    truncLenF = experiment.parameters["truncLenF"]
    if truncLenF is None:
        truncLenF = determineTruncLen(summarySample, forward = True)
        logging.info(f">> [Microbiome analysis] Automatic truncLen for forward reads: {truncLenF}")

    truncLenR = experiment.parameters["truncLenR"]
    if truncLenR is None and pairedEnd:
        truncLenR = determineTruncLen(summarySample, forward = False)
        logging.info(f">> [Microbiome analysis] Automatic truncLen for reverse reads: {truncLenR}")

    denoiseOutput = dada2DenoiseSingleSample(
        sample,
        sampleOutputDir,
        pairedEnd,
        trimLeftF,
        truncLenF,
        trimLeftR,
        truncLenR,
    )

    denoisedSample = ctx_qiime2.createSample(f"{index}-denoise", outputDataset.id, denoiseOutput, experiment, "Step 2: Denoising")

    # Second step:
    # Generate visualization artifacts for the denoised data
    logging.info(">> [Microbiome analysis] Generating visualization")
    denoisedSample.download()
    denoisedSample.unzip()

    visualizationPath = metadataTabulateSample(denoisedSample, sampleOutputDir)
    ctx_qiime2.createSample(f"{index}-metadata-tabulate", outputDataset.id, visualizationPath, experiment, "Step 2: Denoising")

    # Third step:
    # Summarize how many sequences are associated with each sample and with each feature,
    # histograms of those distributions, and some related summary statistics
    logging.info(">> [Microbiome analysis] Creating summarization")
    metadataPath = Path(importedSample.path) / experiment.parameters["metadataFileName"]
    featureTableSummaryPath = featureTableSummarizeSample(denoisedSample, metadataPath, sampleOutputDir)

    ctx_qiime2.createSample(f"{index}-feature-table-summarize", outputDataset.id, featureTableSummaryPath, experiment, "Step 2: Denoising")

    # Fourth step:
    # Provide a mapping of feature IDs to sequences,
    # and provide links to easily BLAST each sequence against the NCBI nt database
    logging.info(">> [Microbiome analysis] Creating mapping file between feature IDs and sequences")
    featureTableMapPath = featureTableTabulateSeqsSample(denoisedSample, sampleOutputDir)
    ctx_qiime2.createSample(f"{index}-feature-table-tabulate-seqs", outputDataset.id, featureTableMapPath, experiment, "Step 2: Denoising")


def main(experiment: Experiment[CustomDataset]):
    dataset = experiment.dataset
    dataset.download()

    demuxSamples = ctx_qiime2.getDemuxSamples(experiment.dataset)
    if len(demuxSamples) == 0:
        raise ValueError(">> [Workspace] Dataset has 0 demultiplexed samples")

    outputDir = folder_manager.createTempFolder("qiime_output")
    outputDataset = CustomDataset.createDataset(
        f"{experiment.id} - Step 2: Denoise",
        experiment.spaceId
    )

    if outputDataset is None:
        raise ValueError(">> [Workspace] Failed to create output dataset")

    for sample in demuxSamples:
        sample.unzip()

        index = ctx_qiime2.sampleNumber(sample)

        importedSample = dataset.getSample(f"{index}-import")
        if importedSample is None:
            raise ValueError(f">> [Microbiome analysis] Imported sample not found")

        importedSample.unzip()

        summarySample = dataset.getSample(f"{index}-summary")
        if summarySample is None:
            raise ValueError(f">> [Microbiome analysis] Summary sample not found")

        summarySample.unzip()

        processSample(
            index,
            sample,
            importedSample,
            summarySample,
            experiment,
            outputDataset,
            outputDir
        )


if __name__ == "__main__":
    initializeProject(main)

import logging

from coretex import Experiment, CustomSample, CustomDataset, folder_manager
from coretex.project import initializeProject

from model import generate


def main(experiment: Experiment) -> None:
    outputDataset = CustomDataset.createDataset(f"{experiment.id} - Generated Images", experiment.spaceId)
    if outputDataset is None:
        raise RuntimeError(">> [Stable Diffusion] Failed to create output dataset")

    outputDir = folder_manager.createTempFolder("images")
    for i in range(experiment.parameters["numOfImages"]):
        logging.info(f">> [Stable Diffusion] Generating image {i + 1}")
        imagePath = outputDir / f"image{i + 1}.png"
        generate(
            imagePath,
            experiment.parameters["prompt"],
            experiment.parameters["height"],
            experiment.parameters["width"],
            experiment.parameters["steps"],
            experiment.parameters["seed"]
        )

        if CustomSample.createCustomSample(imagePath.stem, outputDataset.id, imagePath) is None:
            raise RuntimeError(f">> [Stable Diffusion] Failed to upload {imagePath.name} to coretex")

        logging.info(f">> [Stable Diffusion] Uploaded {imagePath.name} to dataset {outputDataset.name}")

    logging(">> [Stable Diffusion] All images have been generated and uploaded to coretex successfully")


if __name__ == "__main__":
    initializeProject(main)

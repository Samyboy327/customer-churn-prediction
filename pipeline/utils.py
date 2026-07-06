import logging

from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo

# ==========================================================
# Logging
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ==========================================================
# Create PropertyFile
# ==========================================================

def create_evaluation_report():
    """
    Creates the PropertyFile that exposes
    evaluation.json to the SageMaker Pipeline.
    """

    logger.info("Creating PropertyFile...")

    return PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json"
    )


# ==========================================================
# Read Accuracy Metric
# ==========================================================

def get_accuracy_condition(
    evaluation_step,
    evaluation_report,
    threshold
):
    """
    Reads the accuracy metric from evaluation.json
    and creates the approval condition.
    """

    logger.info("Creating accuracy condition...")

    return ConditionGreaterThanOrEqualTo(

        left=JsonGet(

            step_name=evaluation_step.name,

            property_file=evaluation_report,

            json_path="classification_metrics.accuracy.value"

        ),

        right=threshold
    )
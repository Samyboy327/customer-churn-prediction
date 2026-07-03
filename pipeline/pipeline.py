"""
Amazon SageMaker Pipeline
Customer Churn Prediction MLOps Project
"""

import boto3
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput

from sagemaker.session import Session
from sagemaker.workflow.pipeline_context import PipelineSession

from sagemaker.workflow.functions import (
    JsonGet,
    Join,
)

# ==========================================================
# SageMaker Processing & Training
# ==========================================================

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn



# ==========================================================
# Workflow Parameters
# ==========================================================

from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterFloat,
    ParameterString,
)

# ==========================================================
# Workflow Steps
# ==========================================================

from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)

from sagemaker.workflow.pipeline import Pipeline

# ==========================================================
# Model Registration
# ==========================================================

from sagemaker.workflow.properties import PropertyFile
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)

from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import (
    ConditionGreaterThanOrEqualTo,
)

from sagemaker.workflow.functions import JsonGet

from sagemaker.workflow.step_collections import RegisterModel

# ==========================================================
# Project Files
# ==========================================================

from pipeline.config import *
from pipeline.utils import (
    create_evaluation_report,
    get_accuracy_condition,
)

# ==========================================================
# AWS Session
# ==========================================================

boto_session = boto3.Session(
    region_name=AWS_REGION
)

sagemaker_session = Session(
    boto_session=boto_session
)

pipeline_session = PipelineSession(
    boto_session=boto_session
)

role = ROLE

# ==========================================================
# Pipeline Parameters
# ==========================================================

processing_instance_count = ParameterInteger(
    name="ProcessingInstanceCount",
    default_value=INSTANCE_COUNT,
)

processing_instance_type = ParameterString(
    name="ProcessingInstanceType",
    default_value=PROCESSING_INSTANCE_TYPE,
)

training_instance_type = ParameterString(
    name="TrainingInstanceType",
    default_value=TRAINING_INSTANCE_TYPE,
)

model_approval_status = ParameterString(
    name="ModelApprovalStatus",
    default_value="PendingManualApproval",
)

accuracy_threshold = ParameterFloat(
    name="AccuracyThreshold",
    default_value=ACCURACY_THRESHOLD,
)

# ==========================================================
# Processing Processor
# ==========================================================

script_processor = SKLearnProcessor(

    framework_version=SKLEARN_FRAMEWORK_VERSION,

    role=role,

    instance_type=processing_instance_type,

    instance_count=processing_instance_count,

    sagemaker_session=pipeline_session,

)

# ==========================================================
# Training Estimator
# ==========================================================

estimator = SKLearn(

    entry_point=TRAIN_SCRIPT,

    source_dir=SOURCE_DIR,

    role=role,

    framework_version=SKLEARN_FRAMEWORK_VERSION,

    py_version=PYTHON_VERSION,

    instance_count=INSTANCE_COUNT,

    instance_type=training_instance_type,

    output_path=MODEL_OUTPUT_URI,

    sagemaker_session=pipeline_session,

    dependencies=[
        "requirements.txt"
    ]

)

# ==========================================================
# Processing Step
# ==========================================================

processing_step = ProcessingStep(

    name="CustomerChurnPreprocessing",

    processor=script_processor,

    code=PREPROCESSING_SCRIPT,

    inputs=[

        sagemaker.processing.ProcessingInput(

            source=RAW_DATA_URI,

            destination="/opt/ml/processing/input",

            input_name="raw_data",

        )

    ],

    outputs=[

        sagemaker.processing.ProcessingOutput(

            output_name="train",

            source="/opt/ml/processing/train",

        ),

        sagemaker.processing.ProcessingOutput(

            output_name="test",

            source="/opt/ml/processing/test",

        ),

    ],

)

# ==========================================================
# Training Step
# ==========================================================

training_step = TrainingStep(

    name="CustomerChurnTraining",

    estimator=estimator,

    inputs={

        "train": sagemaker.inputs.TrainingInput(

            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
                "train"
            ].S3Output.S3Uri,

            content_type="text/csv",

        )

    },

)

# ==========================================================
# Evaluation Property File
# ==========================================================

evaluation_report = PropertyFile(

    name="EvaluationReport",

    output_name="evaluation",

    path=EVALUATION_FILE,

)

# ==========================================================
# Evaluation Step
# ==========================================================

evaluation_step = ProcessingStep(

    name="CustomerChurnEvaluation",

    processor=script_processor,

    code=EVALUATE_SCRIPT,

    property_files=[evaluation_report],

    inputs=[

        ProcessingInput(

            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,

            destination="/opt/ml/processing/model",

        ),

        ProcessingInput(

            source=processing_step.properties.ProcessingOutputConfig.Outputs[
                "test"
            ].S3Output.S3Uri,

            destination="/opt/ml/processing/test",

        ),

    ],

    outputs=[

        ProcessingOutput(

            output_name="evaluation",

            source="/opt/ml/processing/evaluation",

        )

    ],

)

# ==========================================================
# Model Metrics
# ==========================================================

model_metrics = ModelMetrics(

    model_statistics=MetricsSource(

        s3_uri=Join(

            on="/",

            values=[

                evaluation_step.properties.ProcessingOutputConfig.Outputs[
                    "evaluation"
                ].S3Output.S3Uri,

                EVALUATION_FILE,

            ],

        ),

        content_type="application/json",

    )

)

# ==========================================================
# Accuracy Condition
# ==========================================================

accuracy_condition = ConditionGreaterThanOrEqualTo(

    left=JsonGet(

        step_name=evaluation_step.name,

        property_file=evaluation_report,

        json_path="metrics.accuracy",

    ),

    right=accuracy_threshold,

)

# ==========================================================
# Register Model Step
# ==========================================================

register_model_step = RegisterModel(

    name="RegisterCustomerChurnModel",

    estimator=estimator,

    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,

    content_types=[

        "text/csv"

    ],

    response_types=[

        "text/csv"

    ],

    inference_instances=[

        "ml.m5.large"

    ],

    transform_instances=[

        "ml.m5.large"

    ],

    model_package_group_name=MODEL_PACKAGE_GROUP_NAME,

    approval_status=model_approval_status,

    model_metrics=model_metrics,

)

# ==========================================================
# Condition Step
# ==========================================================

condition_step = ConditionStep(

    name="CheckModelAccuracy",

    conditions=[

        accuracy_condition

    ],

    if_steps=[

        register_model_step

    ],

    else_steps=[],

)

# ==========================================================
# Build Pipeline
# ==========================================================

pipeline = Pipeline(

    name=PIPELINE_NAME,

    parameters=[

        processing_instance_count,

        processing_instance_type,

        training_instance_type,

        accuracy_threshold,

        model_approval_status,

    ],

    steps=[

        processing_step,

        training_step,

        evaluation_step,

        condition_step,

    ],

    sagemaker_session=pipeline_session,

)

# ==========================================================
# Execute Pipeline
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Creating / Updating SageMaker Pipeline...")
    print("=" * 60)

    pipeline.upsert(

        role_arn=ROLE

    )

    print("Pipeline successfully created.")

    execution = pipeline.start()

    print("=" * 60)
    print("Pipeline execution started.")
    print("=" * 60)

    print(f"Execution ARN: {execution.arn}")



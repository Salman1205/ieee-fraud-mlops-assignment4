from kfp import compiler, dsl


@dsl.component(base_image="python:3.10")
def data_ingestion_op():
    return "ingestion_done"


@dsl.component(base_image="python:3.10")
def data_validation_op():
    return "validation_done"


@dsl.component(base_image="python:3.10")
def data_preprocessing_op():
    return "preprocessing_done"


@dsl.component(base_image="python:3.10")
def feature_engineering_op():
    return "feature_engineering_done"


@dsl.component(base_image="python:3.10")
def model_training_op():
    return "training_done"


@dsl.component(base_image="python:3.10")
def model_evaluation_op() -> float:
    # Real project should read metrics artifact. Placeholder for gate wiring.
    return 0.86


@dsl.component(base_image="python:3.10")
def conditional_deploy_op(approved: bool):
    if approved:
        print("Deploying model service...")
    else:
        print("Skipping deployment due to threshold gate.")


@dsl.pipeline(name="fraud-detection-pipeline")
def fraud_pipeline(recall_threshold: float = 0.80):
    ingest = data_ingestion_op().set_retry(2)
    validate = data_validation_op().after(ingest).set_retry(2)
    preprocess = data_preprocessing_op().after(validate).set_retry(2)
    features = feature_engineering_op().after(preprocess).set_retry(2)
    train = model_training_op().after(features).set_retry(1)
    evaluate = model_evaluation_op().after(train).set_retry(1)

    with dsl.If(evaluate.output >= recall_threshold):
        conditional_deploy_op(approved=True).after(evaluate)

    with dsl.Else():
        conditional_deploy_op(approved=False).after(evaluate)


if __name__ == "__main__":
    compiler.Compiler().compile(fraud_pipeline, package_path="artifacts/fraud_pipeline.yaml")

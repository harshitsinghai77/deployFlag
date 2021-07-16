import copy
import datetime
import json
import random
import uuid

import pandas as pd

from deployflag.contrib.ocave.tasks import Pipeline
from deployflag.core.utils import load_json

base_path = "tests/testingFiles"

fname_daily = "observability_daily"
fname_daily_frequency_detected_increase = (
    "observability_daily_frequency_detected_increase"
)
fname_daily_frequency_detected_decrease = (
    "observability_daily_frequency_detected_decrease"
)
fname_daily_frequency_detected_mixed = "observability_daily_frequency_detected_mixed"

fname_historical = "observability_historical"
fname_historical_frequency_increase = "observability_historical_frequency_increase"
fname_historical_frequency_decrease = "observability_historical_frequency_decrease"
fname_historical_frequency_mixed = "observability_historical_frequency_mixed"

special_shortcode = [
    "PYL-R1710",
    "PYL-W0107",
    "PYL-W0212",
    "PYL-W0221",
    "PYL-W0223",
    "PYL-W0511",
]


def save_file(fname, dictionary):
    """Write output as json file."""
    with open(f"{base_path}/{fname}.json", "w") as json_file:
        json.dump(dictionary, json_file)


def run_deployflag(fname, repo_id=1234):
    """Run training for deployflag using file data."""
    fname = f"{base_path}/{fname}.json"
    data_dump = load_json(fname)
    df_daily = pd.DataFrame(data_dump["analysis_run"])
    df_historical = pd.DataFrame(data_dump["historical_run"])

    pipeline = Pipeline(
        repo_id=repo_id, daily_issue=df_daily, historical_issue=df_historical
    )
    pipeline.set_historical_shortcodes()
    pipeline.preprocessing()
    pipeline.combine_and_merge_preprocessed_results()
    pipeline.training()
    pipeline.inference()

    inference_result = pipeline.inference_results
    return inference_result


def generate_shortcode(num=1):
    """Generate random shortcode."""
    shortcodes = [
        "PYL-R1710",
        "PYL-W0107",
        "PYL-W0212",
        "PYL-W0221",
        "PYL-W0223",
        "PYL-W0511",
        "PYL-W0611",
        "PYL-W0613",
        "PYL-W0621",
        "PYL-W0622",
        "PYL-W0703",
        "PYL-W1402",
        "PYL-W1510",
        "PYL-W5104",
        "TCV-001",
    ]

    return [random.choice(shortcodes) for _ in range(num)]


def generate_initial_data(
    repo_id,
    daily_shortcodes,
    historical_shortcodes,
    total_daily_data=8000,
    total_historical_data=7000,
    total_branches=11,
    total_base_oids=20,
    daily_frequency_detected_range=300,
    historical_frequency_range=300,
    include_daily_frequency_detected=True,
    include_historical_frequency=True,
):
    """
    Generate data based on conditions.

    Input:
    total_daily_data: Number of data points for daily_run
    total_historical_data: Number of datapoints for historical_run
    include_daily_frequency_detected: if True, add random `frequency_detected` in daily_run.
    include_historical_frequency: if True, add random `frequency` in historical_run

    Output:
    dict -> Randomly generated data with keys `analysis_run` and `historical_run`
    """

    branch_names = [f"feature-{i}" for i in range(1, total_branches)]
    issue_types = [
        "Bug Risk",
        "Documentation",
        "Security",
        "Style",
        "Anti-pattern",
        "Performance",
    ]
    severity_types = ["major", "critical", "minor"]
    base_oids = [str(uuid.uuid4())[:7] for i in range(total_base_oids)]

    if not daily_shortcodes or not historical_shortcodes:
        print("No shortcodes found.")
        return None

    if (
        len(daily_shortcodes) != total_daily_data
        or len(historical_shortcodes) != total_historical_data
    ):
        print("Not enough shortcodes found.")
        return None

    daily_run = []
    for i in range(total_daily_data):
        daily_run_data = {
            "repository_id": repo_id,
            "run_id": str(uuid.uuid4()),
            "branch_name": random.choice(branch_names),
            "base_oid": random.choice(base_oids),
            "commit_oid": str(uuid.uuid4())[:7],
            "check_id": random.randrange(400, 800),
            "created_at": (
                datetime.datetime.now()
                + datetime.timedelta(days=random.randrange(1, 365))
            ).isoformat(),
            "shortcode": daily_shortcodes[i],
            "issue_type": random.choice(issue_types),
            "severity": random.choice(severity_types),
            "frequency_resolved": random.randrange(1, 150),
        }

        if include_daily_frequency_detected:
            daily_run_data["frequency_detected"] = random.randrange(
                0, daily_frequency_detected_range
            )

        daily_run.append(daily_run_data)

    historical_run = []
    for i in range(total_historical_data):
        historical_data = {
            "run_id": str(uuid.uuid4()),
            "base_oid": str(uuid.uuid4())[:7],
            "timestamp": (
                datetime.datetime.now()
                + datetime.timedelta(days=random.randrange(1, 365))
            ).isoformat(),
            "shortcode": historical_shortcodes[i],
        }

        if include_historical_frequency:
            historical_data["frequency"] = random.randrange(
                0, historical_frequency_range
            )

        historical_run.append(historical_data)

    final_data = {
        "analysis_run": daily_run,
        "historical_run": historical_run,
    }
    return final_data


def get_results_from_inference(inference_results):
    """Create a dict using `shortcode` and `weight`."""
    inferenct_dict = {}
    for result in inference_results:
        issue_shortcode, issue_weight = result.values()
        inferenct_dict[issue_shortcode] = issue_weight
    return inferenct_dict


def calculate_percentage_increase(new_value, original_value):
    """Percentage increase = Increase ÷ Original Number × 100."""
    percentage_increase = ((new_value - original_value) / original_value) * 100
    return round(percentage_increase, 2)


def generate_data_to_monitor_daily_frequency_detected():
    """
    Effect on frequency_detected while keeping other things constant.

    Generate initial_data, keeping initial_data constant generate and save dataset for 3 cases:
    1. Randomly generated frequency_detected.
    2. Decrease frequency_detected.
    3. Increase frequency_detected.
    """
    repo_id = 1234
    total_daily_data = 8000
    total_historical_data = 7000
    frequency_detected_range = 400
    historical_frequency_range = 400
    daily_shortcodes = generate_shortcode(num=total_daily_data)
    historical_shortcodes = generate_shortcode(num=total_historical_data)

    initial_data = generate_initial_data(
        repo_id=repo_id,
        total_daily_data=total_daily_data,
        total_historical_data=total_historical_data,
        daily_shortcodes=daily_shortcodes,
        historical_shortcodes=historical_shortcodes,
        historical_frequency_range=historical_frequency_range,
        include_historical_frequency=True,
        include_daily_frequency_detected=False,
    )

    # set random value for frequency_detected between a range.
    for data in initial_data["analysis_run"]:
        data["frequency_detected"] = random.randrange(
            1, frequency_detected_range)
    save_file(fname_daily, initial_data)

    # for special shortcodes, increase frequency_detected by a constant.
    increase_frequency_detected = copy.deepcopy(initial_data)
    for data in increase_frequency_detected["analysis_run"]:
        if data["shortcode"] in special_shortcode:
            data["frequency_detected"] += 80
    save_file(fname_daily_frequency_detected_increase,
              increase_frequency_detected)

    # for special shortcodes, decrease frequency_detected by a constant
    decrease_frequency_detected = copy.deepcopy(initial_data)
    for data in decrease_frequency_detected["analysis_run"]:
        if data["shortcode"] in special_shortcode:
            new_value = data["frequency_detected"] - 40
            data["frequency_detected"] = max(0, new_value)
    save_file(fname_daily_frequency_detected_decrease,
              decrease_frequency_detected)

    # for few shortcodes, increase and decrease frequency_detected by a constant.
    mixed_frequency_detected = copy.deepcopy(initial_data)
    for data in mixed_frequency_detected["analysis_run"]:
        # increase frequency_detected for the first 3 shortcodes
        if data["shortcode"] in special_shortcode[:3]:
            data["frequency_detected"] += 70

        # decrease frequency_detected for the last 3 shortcodes
        if data["shortcode"] in special_shortcode[3:]:
            new_value = data["frequency_detected"] - 40
            data["frequency_detected"] = max(0, new_value)
    save_file(fname_daily_frequency_detected_mixed, mixed_frequency_detected)


def generate_data_to_monitor_historical_frequency():
    """
    Effect on historical frequency while keep other things constant.

    Generate initial_data, keeping initial_data constant generate and save dataset for 3 cases:
    1. Randomly generated frequency.
    2. Decrease frequency.
    3. Increase frequency.
    """
    repo_id = 1234
    total_daily_data = 8000
    total_historical_data = 7000
    frequency_detected_range = 400
    daily_shortcodes = generate_shortcode(num=total_daily_data)
    historical_shortcodes = generate_shortcode(num=total_historical_data)

    initial_data = generate_initial_data(
        repo_id=repo_id,
        total_daily_data=total_daily_data,
        total_historical_data=total_historical_data,
        daily_shortcodes=daily_shortcodes,
        historical_shortcodes=historical_shortcodes,
        include_historical_frequency=False,
        include_daily_frequency_detected=True,
    )

    # set random value for frequency between a range
    for data in initial_data["historical_run"]:
        data["frequency"] = random.randrange(1, frequency_detected_range)
    save_file(fname_historical, initial_data)

    # for special shortcodes, increase frequency by a constant.
    increase_frequency_detected = copy.deepcopy(initial_data)
    for data in increase_frequency_detected["historical_run"]:
        if data["shortcode"] in special_shortcode:
            data["frequency"] += 80
    save_file(fname_historical_frequency_increase, increase_frequency_detected)

    # for special shortcodes, decrease frequency by a constant
    decrease_frequency_detected = copy.deepcopy(initial_data)
    for data in decrease_frequency_detected["historical_run"]:
        if data["shortcode"] in special_shortcode:
            new_value = data["frequency"] - 40
            data["frequency"] = max(0, new_value)
    save_file(fname_historical_frequency_decrease, decrease_frequency_detected)

    # for few shortcodes, increase and decrease frequency_detected by a constant.
    mixed_frequency_detected = copy.deepcopy(initial_data)
    for data in mixed_frequency_detected["historical_run"]:
        # increase frequency_detected for the first 3 shortcodes
        if data["shortcode"] in special_shortcode[:3]:
            data["frequency"] += 80

        # decrease frequency_detected for the last 3 shortcodes
        if data["shortcode"] in special_shortcode[3:]:
            new_value = data["frequency"] - 40
            data["frequency"] = max(0, new_value)
    save_file(fname_historical_frequency_mixed, mixed_frequency_detected)


def generate_data_to_monitor_shortcodes():
    """
    Effect on shortcode when creating more data points for few specific shortcodes.

    1. Generate and save initial_data.
    2. Generate more data points for few special shortcodeas.
    3. Combine and save data generated in step1 and step2.
    4. Compare the results.
    """

    # create initial data
    repo_id = 1234
    total_daily_data = 2000
    total_historical_data = 3000
    daily_shortcodes = generate_shortcode(num=total_daily_data)
    historical_shortcodes = generate_shortcode(num=total_historical_data)

    initial_data = generate_initial_data(
        repo_id=repo_id,
        total_daily_data=total_daily_data,
        total_historical_data=total_historical_data,
        daily_shortcodes=daily_shortcodes,
        historical_shortcodes=historical_shortcodes,
        daily_frequency_detected_range=200,
        historical_frequency_range=200,
        include_historical_frequency=True,
        include_daily_frequency_detected=True,
    )

    fname = "observability_shortcode_default"
    save_file(fname, initial_data)

    # create more data points for special shortcodes
    number_new_datapoints = 2000
    random_special_shortcode = random.choices(
        special_shortcode, k=number_new_datapoints
    )

    new_data = generate_initial_data(
        repo_id=repo_id,
        total_daily_data=number_new_datapoints,
        total_historical_data=number_new_datapoints,
        daily_shortcodes=random_special_shortcode,
        historical_shortcodes=random_special_shortcode,
        daily_frequency_detected_range=200,
        historical_frequency_range=200,
        include_historical_frequency=True,
        include_daily_frequency_detected=True,
    )

    final_data = {
        "analysis_run": initial_data["analysis_run"] + new_data["analysis_run"],
        "historical_run": initial_data["historical_run"] + new_data["historical_run"],
    }

    assert len(final_data["analysis_run"]) == len(initial_data["analysis_run"]) + len(
        new_data["analysis_run"]
    )

    assert len(final_data["historical_run"]) == len(
        initial_data["historical_run"]
    ) + len(new_data["historical_run"])

    fname = "observability_shortcode_v1"
    save_file(fname, final_data)


def test_observability_for_daily_frequency_detected():
    """
    Test observaility on `frequency_detected` for daily_run.

    Experiment: To compare and observe the result of original (randomly generated fd) dataset
    with subsequently increasing and decreasing `frequency_detected` dataset.

    Result: The `issue_weight` generally had a minor increase when compared original (random)
    with other cases. For some shortcodes, there was a very minimal decrease in the weight.
    And few outliers decreasing the issue_weight significantly.
    """

    # For baseline score
    fname = fname_daily
    repo_id = 12345

    inference_results = run_deployflag(fname=fname, repo_id=repo_id)
    original_result = get_results_from_inference(inference_results)

    # For frequency_detected increase
    fname = fname_daily_frequency_detected_increase
    inference_results_increase = run_deployflag(
        fname=fname,
        repo_id=repo_id,
    )
    fd_increase_result = get_results_from_inference(inference_results_increase)

    print(
        "Percentage increase for increase in frequency detected for special shortcodes"
    )
    for key in special_shortcode:
        perecentage_increase = calculate_percentage_increase(
            fd_increase_result[key], original_result[key]
        )
        assert perecentage_increase < 0
        print("percentage_increase ", key, round(perecentage_increase, 2))

    # For frequency_detected_decrease
    fname = fname_daily_frequency_detected_decrease
    inference_results_decrease = run_deployflag(
        fname=fname,
        repo_id=repo_id,
    )
    fd_decrese_result = get_results_from_inference(inference_results_decrease)

    print(
        "Percentage increase for decrease in frequency detected for special shortcodes"
    )
    for key in special_shortcode:
        perecentage_increase = calculate_percentage_increase(
            fd_decrese_result[key], original_result[key]
        )
        print("percentage_increase ", key, round(perecentage_increase, 2))
        assert perecentage_increase > -1

    # For mixed frequency_detected_
    fname = fname_daily_frequency_detected_mixed
    inference_results_mixed = run_deployflag(
        fname=fname,
        repo_id=repo_id,
    )
    fd_mixed = get_results_from_inference(inference_results_mixed)
    print("Percentage increase for mixed frequency detected for special shortcodes")
    for key in special_shortcode:
        perecentage_increase = calculate_percentage_increase(
            fd_mixed[key], original_result[key]
        )
        if key in special_shortcode[:3]:
            assert perecentage_increase < 0
        else:
            assert perecentage_increase > 0
        print("percentage_increase ", key, round(perecentage_increase, 2))


def test_observability_for_historical_frequency():
    """
    Test observaility on `frequency` for historical_run.

    Experiment: To compare and observe the result of original (randomly generated frequency)
    dataset with subsequently increasing and decreasing `frequency` dataset.

    Result: The `issue_weight` significantly decreased when compared original (random generated)
    with other cases. For some shortcodes, there was a very minimal increase in the weight.
    """

    fname = fname_historical
    repo_id = 12345

    inference_results = run_deployflag(fname=fname, repo_id=repo_id)
    original_result = get_results_from_inference(inference_results)

    # Frequency decreased
    fname = fname_historical_frequency_decrease
    inference_results_decrease = run_deployflag(
        fname=fname,
        repo_id=repo_id,
    )
    fd_decrese_result = get_results_from_inference(inference_results_decrease)

    print("Percentage increase for decrease in frequency for special shortcodes")
    for key in special_shortcode:
        perecentage = calculate_percentage_increase(
            fd_decrese_result[key], original_result[key]
        )
        assert perecentage > 0
        print("percentage_increase ", key, round(perecentage, 2))

    # Frequency increased
    fname = fname_historical_frequency_increase
    inference_results_increase = run_deployflag(
        fname=fname,
        repo_id=repo_id,
    )
    fd_increase_result = get_results_from_inference(inference_results_increase)

    print("Percentage increase for increase fd")
    for key in special_shortcode:
        percentage = calculate_percentage_increase(
            fd_increase_result[key], original_result[key]
        )
        print("percentage_increase ", key, round(percentage, 2))
        assert perecentage > 0

    # For mixed frequency_detected_
    fname = fname_historical_frequency_mixed
    inference_results_mixed = run_deployflag(
        fname=fname,
        repo_id=repo_id,
    )
    fd_mixed = get_results_from_inference(inference_results_mixed)
    print("Percentage increase for mixed frequency detected for special shortcodes")

    for key in special_shortcode:
        perecentage_increase = calculate_percentage_increase(
            fd_mixed[key], original_result[key]
        )
        if key in special_shortcode[:3]:
            assert perecentage_increase < 0
        else:
            assert perecentage_increase >= 0

        print("percentage_increase ", key, round(perecentage_increase, 2))


def test_observability_for_effect_on_shortcode():
    """
    Test observaility for `shortcode`.

    Experiment: To compare and observe the result of original (randomly generated) dataset
    and the data having more datapoints for a particular shortcodes.

    Result: The `issue_weight` increased for the shortcodes having increase number of datapoints. The more the
    number of data points the more the increase in the `issue_weight`.
    """
    fname = "observability_shortcode_default"
    repo_id = 12345

    inference_results = run_deployflag(fname=fname, repo_id=repo_id)
    original_result = get_results_from_inference(inference_results)

    repo_id = 12345
    fname = "observability_shortcode_v1"
    inference_results_v1 = run_deployflag(
        fname=fname,
        repo_id=repo_id,
    )
    result_v1 = get_results_from_inference(inference_results_v1)

    print("Percentage increase for special_shortcode")
    for key in special_shortcode:
        percentage = calculate_percentage_increase(
            result_v1[key], original_result[key])
        assert percentage > 0
        print("percentage_increase ", key, percentage)

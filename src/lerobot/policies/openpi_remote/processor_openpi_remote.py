#!/usr/bin/env python

from typing import Any

from lerobot.policies.openpi_remote.configuration_openpi_remote import OpenPIRemoteConfig
from lerobot.processor import (
    IdentityProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
)
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def make_openpi_remote_pre_post_processors(
    config: OpenPIRemoteConfig,
    dataset_stats: dict | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=[IdentityProcessorStep()],
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=[IdentityProcessorStep()],
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )

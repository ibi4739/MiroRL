# Copyright 2025 MiroMind Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import traceback
from typing import Any, Dict


def extract_exception_details(exception_group) -> Dict[str, Any]:
    """
    Extract detailed exception information from ExceptionGroup

    Args:
        exception_group: ExceptionGroup instance

    Returns:
        Dictionary containing detailed exception information
    """
    details = {
        "message": str(exception_group),
        "type": type(exception_group).__name__,
        "exceptions": [],
        "traceback": traceback.format_exc(),
    }

    # Extract all sub-exceptions
    for i, exc in enumerate(exception_group.exceptions):
        exc_info = {
            "index": i,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        }
        details["exceptions"].append(exc_info)

    return details

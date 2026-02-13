"""
Numeric Validation Guardrail.

Checks that financial figures cited in the generated answer actually
appear in the source documents used to produce it.
"""

import logging
import re
from typing import List

logger = logging.getLogger(__name__)


class NumericValidator:
    """Guardrail that cross-checks cited figures against source documents."""

    @staticmethod
    def apply_guardrail(response_text: str, source_docs: List[str]) -> str:
        """Check that numeric figures cited in the response appear in the sources."""
        # Combine all source text for searching
        combined_sources = " ".join(source_docs).lower()
        
        # Simple extraction of numbers for existence check
        nums = re.findall(r'\d+(?:\.\d+)?', response_text)
        invalid = []
        for n in nums:
            if len(n) > 2 and n not in combined_sources: # Skip small numbers
                invalid.append(n)
        
        if invalid:
            logger.warning(f"Figures not found in sources: {invalid}")
            # We don't necessarily fail here, but we log it.
        
        return response_text

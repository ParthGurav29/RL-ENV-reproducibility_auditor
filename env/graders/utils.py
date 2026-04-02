"""
Shared grader utilities — keyword matching with bidirectional negation guard.
"""

import re


def is_valid_claim(texts: list[str], keywords: list[str]) -> bool:
    """Check if any keyword appears in any text without dismissive negation.

    Negation is checked bidirectionally:
      - 25 chars BEFORE the keyword: only truly dismissive patterns
        (e.g., "already", "skip", "ignore" — NOT bare "not" which
         appears in legitimate violation reports like "does not call X")
      - 25 chars AFTER the keyword: dismissive conclusions
    """
    for text in texts:
        text_lower = text.lower()
        for kw in keywords:
            kw_lower = kw.lower()

            # Locate keyword
            if "cudnn" in kw_lower:
                kw_re = kw_lower.replace("cudnn.", "cudnn ").replace("cudnn ", r"cudnn[.\s_ -]*")
                match = re.search(kw_re, text_lower)
                if not match:
                    continue
                kw_start, kw_end = match.start(), match.end()
            else:
                idx = text_lower.find(kw_lower)
                if idx == -1:
                    continue
                kw_start, kw_end = idx, idx + len(kw_lower)

            # 25-char context windows
            before = text_lower[max(0, kw_start - 25):kw_start]
            after = text_lower[kw_end:min(len(text_lower), kw_end + 25)]

            # Negation BEFORE keyword — only truly dismissive patterns
            neg_before = re.search(
                r"(?:already|no need|skip|skipping|bypass|ignore|don't need|shouldn't)\b",
                before,
            )
            # Dismissive negation AFTER keyword
            neg_after = re.search(
                r"(?:not required|not needed|not necessary|unnecessary|"
                r"irrelevant|not an issue|already (?:set|present|handled|configured))",
                after,
            )

            if neg_before or neg_after:
                continue

            return True
    return False

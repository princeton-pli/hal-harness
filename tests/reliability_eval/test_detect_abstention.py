"""Tests for detect_abstention in run_reliability_eval.py."""

from reliability_eval import detect_abstention


def msg(role, content):
    return {"role": role, "content": content}


class TestDetectAbstentionNoSignals:
    def test_empty_conversation_is_not_abstention(self):
        result = detect_abstention([], [])
        assert result["abstained"] is False
        assert result["abstention_type"] == "none"
        assert result["abstention_strength"] == 0.0
        assert result["num_assistant_messages"] == 0

    def test_normal_assistant_response_is_not_abstention(self):
        history = [msg("assistant", "I have booked your flight to Paris.")]
        result = detect_abstention(
            history, ["search_flights", "book_flight", "confirm"]
        )
        assert result["abstained"] is False
        assert result["abstention_type"] == "none"

    def test_user_messages_are_ignored(self):
        # User message contains an inability pattern — should not trigger abstention.
        history = [
            msg("user", "I cannot complete this without your help"),
            msg("assistant", "Sure, I will help you with that."),
        ]
        result = detect_abstention(history, ["action1", "action2", "action3"])
        assert result["abstained"] is False
        assert result["num_assistant_messages"] == 1

    def test_system_messages_are_ignored(self):
        history = [
            msg("system", "I cannot proceed beyond this point."),
            msg("assistant", "Task completed successfully."),
        ]
        result = detect_abstention(history, ["a", "b", "c"])
        assert result["abstained"] is False


class TestDetectAbstentionInability:
    def test_i_cannot_triggers_inability(self):
        history = [msg("assistant", "I cannot access external systems.")]
        result = detect_abstention(history, ["a", "b", "c"])
        assert result["abstained"] is True
        assert result["abstention_type"] == "inability"

    def test_i_am_unable_to_triggers_inability(self):
        history = [msg("assistant", "I am unable to complete this task.")]
        result = detect_abstention(history, ["a", "b", "c"])
        assert result["abstained"] is True
        assert result["abstention_type"] == "inability"

    def test_i_cant_triggers_inability(self):
        history = [msg("assistant", "Sorry, I can't do that for you.")]
        result = detect_abstention(history, ["a", "b", "c"])
        assert result["abstained"] is True

    def test_inability_score_is_10(self):
        history = [msg("assistant", "I cannot do this.")]
        result = detect_abstention(history, ["a", "b", "c"])
        assert result["scores_by_type"]["inability"] == 1.0

    def test_evidence_is_populated(self):
        history = [msg("assistant", "I cannot access that file.")]
        result = detect_abstention(history, ["a", "b", "c"])
        assert len(result["evidence"]) >= 1
        assert "[inability]" in result["evidence"][0]


class TestDetectAbstentionRefusal:
    def test_cannot_proceed_triggers_refusal(self):
        history = [msg("assistant", "I cannot proceed with this request.")]
        result = detect_abstention(history, ["a", "b", "c"])
        assert result["abstained"] is True
        assert result["abstention_type"] == "refusal"

    def test_i_must_stop_triggers_refusal(self):
        history = [msg("assistant", "I must stop here for safety reasons.")]
        result = detect_abstention(history, ["a", "b", "c"])
        assert result["abstained"] is True

    def test_stopping_here_triggers_refusal(self):
        history = [msg("assistant", "Stopping here as I do not have permission.")]
        result = detect_abstention(history, ["a", "b", "c"])
        assert result["abstained"] is True

    def test_refusal_is_triggered(self):
        # "I must stop" only matches the single refusal pattern r'\bi must (stop|...)\b'.
        history = [msg("assistant", "I must stop here.")]
        result = detect_abstention(history, ["a", "b", "c"])
        assert result["scores_by_type"]["refusal"] == 1.0


class TestDetectAbstentionUncertainty:
    def test_uncertainty_alone_does_not_abstain(self):
        # Single uncertainty signal scores 0.7, which is below the 0.9 threshold
        # (abstention_strength = 0.7/3 = 0.23 < 0.3, and type not inability/refusal)
        history = [msg("assistant", "I'm not sure about the best approach here.")]
        result = detect_abstention(history, ["a", "b", "c"])
        assert result["abstained"] is False
        assert result["scores_by_type"]["uncertainty"] == 0.7

    def test_multiple_uncertainty_signals_can_abstain(self):
        # Two uncertainty signals: 1.4 total → strength 1.4/3 = 0.47 ≥ 0.3
        history = [
            msg("assistant", "I'm not sure about this. I don't know the right answer."),
        ]
        result = detect_abstention(history, ["a", "b", "c"])
        assert result["abstained"] is True
        assert result["abstention_type"] == "uncertainty"


class TestDetectAbstentionClarification:
    def test_clarification_alone_does_not_abstain(self):
        # "I need more information." only matches the single pattern r'\bi need...information\b'.
        # One hit scores 0.5 → strength 0.5/3 = 0.17 < 0.3 → no abstention.
        history = [msg("assistant", "I need more information about that.")]
        result = detect_abstention(history, ["a", "b", "c"])
        assert result["abstained"] is False
        assert result["scores_by_type"]["clarification"] == 0.5

    def test_need_more_information_is_clarification(self):
        history = [msg("assistant", "I need more information to proceed with this.")]
        result = detect_abstention(history, ["a", "b", "c"])
        assert result["scores_by_type"]["clarification"] == 0.5


class TestDetectAbstentionEarlyTermination:
    def test_zero_actions_is_early_termination(self):
        result = detect_abstention([], [])
        assert result["early_termination"] is True

    def test_two_actions_is_early_termination(self):
        result = detect_abstention([], ["a", "b"])
        assert result["early_termination"] is True

    def test_three_actions_is_not_early_termination(self):
        result = detect_abstention([], ["a", "b", "c"])
        assert result["early_termination"] is False


class TestDetectAbstentionEdgeCases:
    def test_non_string_content_does_not_raise(self):
        # Content may be a list (e.g. tool call blocks) rather than a plain string.
        history = [msg("assistant", [{"type": "text", "text": "Done."}])]
        result = detect_abstention(history, ["a", "b", "c"])
        assert result["abstained"] is False

    def test_evidence_capped_at_five(self):
        # Many matching patterns in one message — evidence should be ≤ 5.
        text = (
            "I cannot proceed. I must stop. Stopping here. "
            "I cannot continue. Unable to proceed. I cannot complete."
        )
        history = [msg("assistant", text)]
        result = detect_abstention(history, ["a", "b", "c"])
        assert len(result["evidence"]) <= 5

    def test_abstention_strength_capped_at_one(self):
        # Very high accumulated score should not push strength above 1.0.
        text = " ".join(
            [
                "I cannot do this. I am unable to help. I can't proceed.",
                "I must stop. Stopping here. Unable to continue. Cannot proceed.",
            ]
        )
        history = [msg("assistant", text)]
        result = detect_abstention(history, [])
        assert result["abstention_strength"] <= 1.0

    def test_multiple_assistant_messages_accumulate(self):
        history = [
            msg("assistant", "I'm not sure about this."),
            msg("assistant", "I don't know the answer."),
            msg("assistant", "I'm uncertain about the approach."),
        ]
        result = detect_abstention(history, ["a", "b", "c"])
        assert result["num_assistant_messages"] == 3
        assert result["scores_by_type"]["uncertainty"] > 0.7

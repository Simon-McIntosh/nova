from __future__ import annotations

import re
import sys
from datetime import (
    date,
    datetime,
    time
)
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    ClassVar,
    Literal,
    Optional,
    Union
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer
)


metamodel_version = "1.11.0"
version = "1.0.0"


class ConfiguredBaseModel(BaseModel):
    model_config = ConfigDict(
        serialize_by_alias = True,
        validate_by_name = True,
        validate_assignment = True,
        validate_default = True,
        extra = "forbid",
        arbitrary_types_allowed = True,
        use_enum_values = True,
        strict = False,
    )





class LinkMLMeta(RootModel):
    root: dict[str, Any] = {}
    model_config = ConfigDict(frozen=True)

    def __getattr__(self, key:str):
        return getattr(self.root, key)

    def __getitem__(self, key:str):
        return self.root[key]

    def __setitem__(self, key:str, value):
        self.root[key] = value

    def __contains__(self, key:str) -> bool:
        return key in self.root


linkml_meta = LinkMLMeta({'default_prefix': 'correction',
     'default_range': 'string',
     'description': 'A diagnostic correction is a function of pulse range, not a '
                    'constant.  A channel holds one calibration state over a run '
                    'of pulses, steps, holds, and sometimes steps back; a single '
                    'number fitted across such a step is an average of discrete '
                    'states weighted by pulse count, describing no pulse and '
                    'moving when the pulse selection moves.  This schema therefore '
                    'scopes every correction to a validity interval and attaches '
                    'the evidence that established it, so a consumer reading one '
                    'channel on one pulse can say which correction applies, what '
                    'warranted it, and whether anything warranted it at all.\n'
                    "One document carries one machine's corrections for one "
                    'diagnostic system.  The read path applies the corrections '
                    'whose status is promoted, in the order the ApplicationStage '
                    'enum ranks, and reports the rest rather than silently '
                    'applying them: a refusal with a reason is worth more than a '
                    'silence.',
     'id': 'https://github.com/Simon-McIntosh/nova/schema/diagnostic-correction',
     'imports': ['linkml:types'],
     'license': 'ITER GIP',
     'name': 'diagnostic-correction',
     'prefixes': {'correction': {'prefix_prefix': 'correction',
                                 'prefix_reference': 'https://github.com/Simon-McIntosh/nova/schema/diagnostic-correction/'},
                  'linkml': {'prefix_prefix': 'linkml',
                             'prefix_reference': 'https://w3id.org/linkml/'}},
     'source_file': 'nova/calibrate/schema/diagnostic_correction.yaml',
     'title': 'Diagnostic Correction Schema'} )

class CorrectionKind(str, Enum):
    """
    What a correction states about a channel.  The kinds are disjoint by the quantity they carry rather than by the fault they describe, because one fault can show up as several kinds: a failed pickup pair is a pair_state on the pulses where the state is resolved and a quality state where it is not.
    """
    gain = "gain"
    """
    Scalar multiplier between what the channel reports and the physical quantity it measures.  Divided out on the read path.
    """
    acquisition_scale = "acquisition_scale"
    """
    Scalar the acquisition chain applied because of a range setting, constant over a block of pulses and quantised onto a declared ladder.  It is not a calibration of the sensor: what is divided out is the ladder rung, never the measured ratio, because the measured ratio also carries whatever the forward description gets wrong about this channel.
    """
    pair_state = "pair_state"
    """
    Which members of a multi-element pickup contributed to the recorded signal over an interval.  A channel whose two elements are averaged before recording halves when one element fails, so the state is an enumerated wiring condition with a nominal multiplier, not a free scalar.
    """
    offset = "offset"
    """
    Additive baseline in the channel's own units, removed before any multiplicative correction.  Usually measured per pulse from the pre-excitation quiet window rather than stored as a set-level constant.
    """
    drift_rate = "drift_rate"
    """
    Slope of an integrator's zero in channel units per second, optionally with a curvature term, removed as a ramp over the pulse.
    """
    exclusion = "exclusion"
    """
    Instruction to drop the channel from a consumer's fit or read, with a stated cause.  Distinct from a quality state: an exclusion is an action, a quality state is a description.
    """
    quality = "quality"
    """
    Enumerated condition of a channel or a named channel group, with a cause and evidence.  Carries the case a fitted correction cannot: a channel that is an outlier against its own neighbours in a way no multiplier explains.
    """
    convention = "convention"
    """
    Global factor and sign sense relating a stored quantity to the quantity a consumer computes, applied at the unit boundary rather than in the per-channel chain.
    """


class CorrectionStatus(str, Enum):
    """
    Whether the read path acts on a correction, and if not, why it is nonetheless in the document.  Recording the refusals beside the promotions is the point: an absent correction cannot be told from an unmeasured one.
    """
    promoted = "promoted"
    """
    Measured, corroborated, and applied by the read path.  Only promoted corrections change a value a consumer reads.
    """
    recorded = "recorded"
    """
    Measured and carried as evidence, but not applied.  The measurement stands; promoting it to the read path is a separate decision with its own gate.
    """
    withheld = "withheld"
    """
    Measured but refused promotion by a stated gate — independent routes that disagree, or a value unstable across the channel's own pulse halves.
    """
    refused = "refused"
    """
    The measurement is not the quantity the kind names, so no value is carried. A block whose step misses every ladder rung is not a range setting, and rounding it onto the nearest rung would assert a setting the ladder does not support.
    """
    superseded = "superseded"
    """
    Replaced by a later correction covering the same pulses, kept for the record.  This is the one status that may share an interval with another, which is why the non-overlap rule is scoped within a status.
    """


class QualityStatus(str, Enum):
    """
    Condition of a channel or channel group.
    """
    good = "good"
    """
    Reads the quantity it names, within its own measured floor.
    """
    suspect = "suspect"
    """
    Behaves unlike its own neighbours in a way no fitted correction removes. Consumers should weight it down or test with and without it; it is not an instruction to drop the channel.
    """
    corrupted = "corrupted"
    """
    Reads a quantity other than the one it names over the interval.
    """
    dead = "dead"
    """
    Carries no signal.
    """


class PairState(str, Enum):
    """
    Which elements of a multi-element pickup contributed to a recorded signal. MAST equilibrium pickups are two coils 180 degrees apart toroidally whose signals are averaged before recording, so the state fixes a nominal multiplier on the recorded amplitude.
    """
    both_members = "both_members"
    """
    Both elements contributing; nominal multiplier 1.0.
    """
    single_member = "single_member"
    """
    One element contributing; nominal multiplier 0.5.
    """
    recovered = "recovered"
    """
    A repaired or rewired pickup summing where the acquisition still divides by the original element count; nominal multiplier 1.5.
    """
    indeterminate = "indeterminate"
    """
    The channel moves between states faster than any interval resolves, so no state and no single multiplier describes the interval.  The observed states belong in candidate_values.
    """


class ApplicationStage(str, Enum):
    """
    The fixed order the read path applies corrections in, lowest rank first.  A correction a consumer cannot order is a correction that will be applied twice or not at all, so the order is part of the schema rather than of one consumer. Additive terms come out before multiplicative ones; the acquisition scale comes out before any sensor gain because the range setting is applied last by the instrument and must therefore be removed first among the multipliers; the pair state precedes the gain because a fitted gain measured across mixed states is itself contaminated by them.
    """
    offset = "offset"
    """
    Subtract the additive baseline.
    """
    drift = "drift"
    """
    Subtract the integrator ramp.
    """
    acquisition_scale = "acquisition_scale"
    """
    Divide out the acquisition range rung.
    """
    pair_state = "pair_state"
    """
    Divide out the pickup state's nominal multiplier.
    """
    gain = "gain"
    """
    Divide out the channel gain.
    """
    convention = "convention"
    """
    Apply the stored-to-computed factor at the unit boundary, after the per-channel chain and never inside it.
    """



class CorrectionSet(ConfiguredBaseModel):
    """
    One machine's corrections for one diagnostic system, versioned as a whole. The set version moves when any correction in it does, so a consumer can pin a calibration the way it pins a code version.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/Simon-McIntosh/nova/schema/diagnostic-correction',
         'tree_root': True})

    machine: str = Field(default=..., description="""Device the corrections describe.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CorrectionSet']} })
    diagnostic_system: str = Field(default=..., description="""Diagnostic the document covers, named as the data-dictionary IDS holding it (magnetics, pf_active, ...).  One document per system keeps a correction's scope legible from its path.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CorrectionSet']} })
    schema_version: str = Field(default=..., description="""Version of this schema the document is written against.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CorrectionSet']} })
    set_version: str = Field(default=..., description="""Monotonic semantic version of the correction set itself, bumped by the author when a value, an interval or a status changes.  Semantic and never a content hash: a hash orders nothing and tells a consumer nothing about compatibility.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CorrectionSet']} })
    generated_by: str = Field(default=..., description="""What produced the document, specifically enough to run again.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CorrectionSet']} })
    generated_at: Optional[date] = Field(default=None, description="""When the document was last written.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CorrectionSet']} })
    description: Optional[str] = Field(default=None, description="""What the set covers and what it deliberately leaves out.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CorrectionSet']} })
    ladders: Optional[list[QuantisationLadder]] = Field(default=None, description="""Quantisation ladders the set's corrections are validated against.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CorrectionSet']} })
    corrections: list[ChannelCorrection] = Field(default=..., description="""Every correction the set carries.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CorrectionSet']} })

    @field_validator('schema_version')
    def pattern_schema_version(cls, v):
        pattern=re.compile(r"^\d+\.\d+\.\d+$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid schema_version format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid schema_version format: {v}"
            raise ValueError(err_msg)
        return v

    @field_validator('set_version')
    def pattern_set_version(cls, v):
        pattern=re.compile(r"^\d+\.\d+\.\d+$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid set_version format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid set_version format: {v}"
            raise ValueError(err_msg)
        return v


class QuantisationLadder(ConfiguredBaseModel):
    """
    The discrete values a quantised correction may take.  Declaring them in the document is what lets validation refuse a free-floating value: a step that misses every rung is a real step the finder measured, but it is not the discrete factor a range setting moves by, and removing it would be a fit rather than a correction.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/Simon-McIntosh/nova/schema/diagnostic-correction'})

    name: str = Field(default=..., description="""Identifier of a ladder, unique within the set.""", json_schema_extra = { "linkml_meta": {'domain_of': ['QuantisationLadder']} })
    kind: CorrectionKind = Field(default=..., description="""What the correction states.""", json_schema_extra = { "linkml_meta": {'domain_of': ['QuantisationLadder', 'ChannelCorrection']} })
    rungs: list[float] = Field(default=..., description="""Values a quantised correction of this kind may take, declared as a hypothesis before any step is classified so that a measurement missing every rung is visible as such instead of being rounded onto one.""", json_schema_extra = { "linkml_meta": {'domain_of': ['QuantisationLadder']} })
    tolerance: float = Field(default=..., description="""Fractional distance from a rung, |value - rung| / rung, inside which a value is said to land on it.  Wide enough to absorb a channel's own few-percent calibration offset riding on the setting, narrow enough that neighbouring rungs cannot be confused.""", json_schema_extra = { "linkml_meta": {'domain_of': ['QuantisationLadder']} })
    basis: Optional[str] = Field(default=None, description="""Why these rungs and not others.""", json_schema_extra = { "linkml_meta": {'domain_of': ['QuantisationLadder']} })


class ChannelCorrection(ConfiguredBaseModel):
    """
    One correction, on one channel or channel group, over one or more validity intervals.  A channel that holds different states over different pulse ranges carries one of these per era.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/Simon-McIntosh/nova/schema/diagnostic-correction'})

    channel: Optional[str] = Field(default=None, description="""Channel the correction applies to, named as the source store names it. Exactly one of channel or channel_group is set.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ChannelCorrection']} })
    channel_group: Optional[str] = Field(default=None, description="""Named set of channels the correction applies to — a probe family, a digitiser group, a whole array failing together.  Exactly one of channel or channel_group is set.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ChannelCorrection']} })
    kind: CorrectionKind = Field(default=..., description="""What the correction states.""", json_schema_extra = { "linkml_meta": {'domain_of': ['QuantisationLadder', 'ChannelCorrection']} })
    status: CorrectionStatus = Field(default=..., description="""Whether the read path applies the correction, and if not, why not.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ChannelCorrection']} })
    value: Optional[float] = Field(default=None, description="""The quantity the read path uses, in the correction's own units: a multiplier for gain, acquisition_scale, pair_state and convention; an additive term for offset; a slope for drift_rate.  Absent when the correction carries no value a consumer may apply, which every non-promoted kind may be.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ChannelCorrection', 'Corroboration']} })
    measured_value: Optional[float] = Field(default=None, description="""The raw measurement behind the value, kept when the two differ.  An acquisition block's measured response ratio carries both the range setting and whatever the forward description gets wrong about the channel; only the rung is a statement about the acquisition, so the read path divides by the rung and the ratio stays here as evidence.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ChannelCorrection']} })
    candidate_values: Optional[list[float]] = Field(default=None, description="""Values the quantity was observed to take when no single one describes the interval, so a consumer can see the size of what is unresolved rather than infer it from a missing value.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ChannelCorrection']} })
    state: Optional[PairState] = Field(default=None, description="""Pickup state over the interval, for a pair_state correction.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ChannelCorrection']} })
    quality_status: Optional[QualityStatus] = Field(default=None, description="""Condition of the channel or group, for a quality correction.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ChannelCorrection']} })
    cause: Optional[str] = Field(default=None, description="""Why the channel is excluded or in this quality state, in mechanism terms.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ChannelCorrection']} })
    unit: Optional[str] = Field(default=None, description="""Unit of value and measured_value, omitted when the quantity is a ratio.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ChannelCorrection', 'Corroboration', 'Uncertainty']} })
    uncertainty: Optional[Uncertainty] = Field(default=None, description="""Interval the value is supported to.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ChannelCorrection', 'Corroboration']} })
    ladder: Optional[str] = Field(default=None, description="""Name of the ladder this correction's value must land on.  Set on a quantised kind so validation can reject a free-floating value.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ChannelCorrection']} })
    validity: list[ValidityInterval] = Field(default=..., description="""Pulse or time spans the correction holds over.  Corrections that the read path would both apply must not overlap, so a value is carried per span and a channel that steps carries one correction per era rather than one averaged constant.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ChannelCorrection']} })
    provenance: Provenance = Field(default=..., description="""What established the correction.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ChannelCorrection']} })
    notes: Optional[str] = Field(default=None, description="""Anything a consumer needs that the structured slots do not carry.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ChannelCorrection', 'ValidityInterval']} })


class ValidityInterval(ConfiguredBaseModel):
    """
    A span of pulses or of time, either end open.  Pulse and time bounds are not mixed in one interval: the two cannot be ordered against each other, so an interval bounded below in pulse and above in time names nothing.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/Simon-McIntosh/nova/schema/diagnostic-correction'})

    pulse_start: Optional[int] = Field(default=None, description="""First pulse the correction holds on; open below when absent.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ValidityInterval']} })
    pulse_end: Optional[int] = Field(default=None, description="""Last pulse the correction holds on; open above when absent.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ValidityInterval']} })
    time_start: Optional[float] = Field(default=None, description="""First time the correction holds at [s]; open below when absent.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ValidityInterval']} })
    time_end: Optional[float] = Field(default=None, description="""Last time the correction holds at [s]; open above when absent.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ValidityInterval']} })
    measured_pulses: Optional[list[int]] = Field(default=None, description="""Pulses the correction was actually measured on, which the span alone does not give.  A block running from pulse 14061 to 19258 may rest on thirty-six of the five thousand pulses between them, and recording the span alone would let a read of pulse 17000 come back measured when nothing measured it.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ValidityInterval']} })
    notes: Optional[str] = Field(default=None, description="""Anything a consumer needs that the structured slots do not carry.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ChannelCorrection', 'ValidityInterval']} })


class Provenance(ConfiguredBaseModel):
    """
    What established a correction.  A value without provenance is indistinguishable from a guess once the session that produced it is over.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/Simon-McIntosh/nova/schema/diagnostic-correction'})

    method: str = Field(default=..., description="""How the quantity was measured, in enough detail to repeat it.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Provenance', 'Corroboration']} })
    evidence_uri: Optional[str] = Field(default=None, description="""Where the evidence lives — a research record, a banked array, or the module the value was mined from.  Repo-relative paths and URLs are both accepted; what matters is that a reader can reach the thing that justified the number.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Provenance', 'Corroboration']} })
    fitted_at: Optional[date] = Field(default=None, description="""When the measurement was made.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Provenance']} })
    fitted_by: Optional[str] = Field(default=None, description="""Who or what made it.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Provenance']} })
    statement: Optional[str] = Field(default=None, description="""What the evidence says, in one sentence.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Provenance', 'Corroboration']} })
    corroborations: Optional[list[Corroboration]] = Field(default=None, description="""Independent routes to the same quantity.  Kept as structured entries rather than prose because which route established a value matters more than how close two routes came, and a route that later moves must be findable.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Provenance']} })


class Corroboration(ConfiguredBaseModel):
    """
    An independent route reaching the same quantity, recorded beside the value it supports rather than in place of it, so a later disagreement between routes is visible instead of hidden by whichever was written.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/Simon-McIntosh/nova/schema/diagnostic-correction'})

    method: str = Field(default=..., description="""How the quantity was measured, in enough detail to repeat it.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Provenance', 'Corroboration']} })
    value: Optional[float] = Field(default=None, description="""The quantity the read path uses, in the correction's own units: a multiplier for gain, acquisition_scale, pair_state and convention; an additive term for offset; a slope for drift_rate.  Absent when the correction carries no value a consumer may apply, which every non-promoted kind may be.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ChannelCorrection', 'Corroboration']} })
    unit: Optional[str] = Field(default=None, description="""Unit of value and measured_value, omitted when the quantity is a ratio.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ChannelCorrection', 'Corroboration', 'Uncertainty']} })
    uncertainty: Optional[Uncertainty] = Field(default=None, description="""Interval the value is supported to.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ChannelCorrection', 'Corroboration']} })
    evidence_uri: Optional[str] = Field(default=None, description="""Where the evidence lives — a research record, a banked array, or the module the value was mined from.  Repo-relative paths and URLs are both accepted; what matters is that a reader can reach the thing that justified the number.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Provenance', 'Corroboration']} })
    statement: Optional[str] = Field(default=None, description="""What the evidence says, in one sentence.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Provenance', 'Corroboration']} })


class Uncertainty(ConfiguredBaseModel):
    """
    The interval a value is supported to, in the value's own units.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/Simon-McIntosh/nova/schema/diagnostic-correction'})

    lower: Optional[float] = Field(default=None, description="""Lower end of the supported interval.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Uncertainty']} })
    upper: Optional[float] = Field(default=None, description="""Upper end of the supported interval.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Uncertainty']} })
    unit: Optional[str] = Field(default=None, description="""Unit of value and measured_value, omitted when the quantity is a ratio.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ChannelCorrection', 'Corroboration', 'Uncertainty']} })


# Model rebuild
# see https://pydantic-docs.helpmanual.io/usage/models/#rebuilding-a-model
CorrectionSet.model_rebuild()
QuantisationLadder.model_rebuild()
ChannelCorrection.model_rebuild()
ValidityInterval.model_rebuild()
Provenance.model_rebuild()
Corroboration.model_rebuild()
Uncertainty.model_rebuild()

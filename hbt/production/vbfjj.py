import functools
from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, set_ak_column
from columnflow.production.util import attach_coffea_behavior

np = maybe_import("numpy")
ak = maybe_import("awkward")

set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses=(
        "Jet.*", "VBFJet.*",
        attach_coffea_behavior,
    ),
    produces={
        "vbfjj.*", "vbfjj_dr",
    },
)
def vbfjj(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](
        events,
        collections={"VBFJet": {"type_name": "Jet"}},
        **kwargs,
    )
    # total number of objects per event
    n_vbfjets = ak.num(events.VBFJet, axis=1)
    # mask to select events with 2 VBF jets
    vbfjet_mask = (n_vbfjets == 2)
    # create the vbf jet pair
    vbfjj = events.VBFJet.sum(axis=1)
    # take only events with the vbf jets
    vbf_events = events.VBFJet[vbfjet_mask]
    # assign vbf jet 1 and jet 2
    vbf1, vbf2 = vbf_events[:, 0], vbf_events[:, 1]
    dr_true = vbf1.delta_r(vbf2)

    padded_delta_r = ak.pad_none(dr_true, len(events.VBFJet), axis=0)
    # Fill the None values with the placeholder EMPTY_FLOAT
    vbfjj_dr = ak.fill_none(padded_delta_r, EMPTY_FLOAT)
    # print("vbfjj_dr:",vbfjj_dr)

    def save_interesting_properties(
        source: ak.Array,
        target_column: str,
        column_values: ak.Array,
        mask: ak.Array[bool],
    ):
        return set_ak_column_f32(
            source,
            target_column,
            ak.where(mask, column_values, EMPTY_FLOAT),
        )
    # write out variables to the corresponding events array, applying certain masks
    events = save_interesting_properties(events, "vbfjj.mass", vbfjj.mass, vbfjet_mask)
    events = save_interesting_properties(events, "vbfjj_dr", vbfjj_dr, vbfjet_mask)
    print("OK END")
    # events = save_interesting_properties(events, "vbfjj.d_eta", hh.eta, dihh_mask)

    # return the events
    return events

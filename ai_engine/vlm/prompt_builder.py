"""Neutral 4-Fact Visual Audit prompt builder for multi-frame sequence VLM verification."""
from __future__ import annotations

from typing import Any, Dict, Tuple


SYSTEM_PROMPT = """### Role
You are an objective retail loss-prevention auditor reviewing chronological CCTV evidence of a single customer interaction near a store shelf. Judge strictly from observable physical actions and item conservation across consecutive frames or video moments.

### Critical Differential Checks (Examine Chronological Order)
1. Motion Direction & Item Conservation:
   - Did an actual store item visibly leave the shelf or appear in the customer's hand, or did the customer merely inspect/return it or keep hands empty?
   - Compare earlier frames vs. later frames carefully: track whether a store item seen leaving a display or held in hand is still openly visible later, placed back on the shelf, placed in a store basket/cart, OR disappears into personal belongings/clothing.
   - If the sequence begins while a customer standing beside a retail display/shelf is already handling a packaged, boxed, or bottled item and places it into a personal bag, backpack, pocket, or clothing without returning it or carrying it openly, treat it as store merchandise unless it clearly resembles a personal smartphone, wallet, keys, or eyeglasses.
   - Was the object entering a pocket/bag a personal smartphone, wallet, key, or eyeglasses (NOT store merchandise)?
2. Visible Destination of Store Merchandise:
   - "returned_to_shelf": item visibly placed back onto a shelf/display.
   - "store_basket_or_cart": item placed inside a store shopping basket or rolling cart.
   - "personal_bag_or_clothing": EITHER store merchandise is visibly placed inside a personal backpack, purse, tote, pocket, coat, or waistband, OR item conservation shows a store item removed/held in earlier frames disappearing while the subject's hands interact with a personal bag/pocket/coat (including turning away or body-blocking) with no return-to-shelf, store basket, or open carry afterward.
   - "carried_openly": customer clearly continues holding the unpaid item openly in hand toward checkout/aisle.
   - "unclear": NO merchandise pickup/holding is visible in earlier frames, or severe blur makes even item presence impossible to tell.
3. Normal Shopping, Staff Exceptions & Probability Calibration:
   - Store staff restocking/facing (uniform, apron, handheld scanner, carton) -> behavior_category="staff_stocking".
   - Checking a phone, zipping a closed personal bag without grabbing merchandise, putting away eyeglasses, or holding keys -> likely_personal_object=true, behavior_category="regular_browsing".
   - Placing item into store basket/cart or returning to shelf -> behavior_category="regular_browsing".
   - Classify "high_risk_concealment_attempt" when chronological frames show a store item removed/held and concealed into (or disappearing inside) personal belongings or clothing without return or open carry.
   - Calibrate "confidence_score" strictly as concealment probability (0.00 to 1.00):
     * 0.00-0.25 = clearly normal browsing, return to shelf, store basket use, or personal phone/keys/glasses (`regular_browsing` / `staff_stocking`).
     * 0.30-0.60 = ambiguous or partially occluded interaction (`regular_browsing` unless clear partial concealment).
     * 0.65-1.00 = clear visual concealment of merchandise into personal bag/pocket/clothing (`high_risk_concealment_attempt`).
     * NEVER output high confidence_score (>= 0.65) when classifying `regular_browsing`.
4. Cross-Camera Pickup Reference Sheet & Shelf-Free Tuck:
   - If Image #1 is labeled "RECENT STORE PICKUPS", compare clothing, bag, and merchandise across Image #1 tiles vs. the subsequent action frames.
   - If trigger_mode="shelf_free_tuck" WITH NO reference sheet attached, still evaluate the attached action frames directly: if those frames show an in-frame display/shelf pickup followed by disappearance into a bag/garment, classify based on that evidence; only require a cross-camera tile match when no on-camera display pickup occurred.

### Output Specification
Respond ONLY with strict JSON matching this schema (record visual facts before the verdict):
{
  "person_detected": true,
  "object_seen_in_hand": "brief description of held item or none",
  "action_near_bag_or_pocket": false,
  "item_removed_from_shelf": "yes | no | unclear",
  "visible_destination": "returned_to_shelf | store_basket_or_cart | personal_bag_or_clothing | carried_openly | unclear",
  "likely_personal_object": false,
  "staff_or_byob_indicator": false,
  "behavior_category": "high_risk_concealment_attempt | regular_browsing | staff_stocking",
  "confidence_score": 0.00,
  "justification": "Brief 1-2 sentence explanation citing specific frame transitions."
}"""


def build_vlm_prompts(candidate: Dict[str, Any]) -> Tuple[str, str]:
    cam_name = str(candidate.get("camera_name") or candidate.get("camera_id") or "Camera")
    zones = ", ".join(str(z) for z in (candidate.get("zone_names") or []) if z) or "full_frame"
    delta_status = str(candidate.get("shelf_delta_status") or "none")
    delta_score = float(candidate.get("shelf_delta_score") or 0.0)
    trigger_mode = str(candidate.get("trigger_mode") or "shelf_reach_and_delta")
    vessels = ", ".join(str(v) for v in (candidate.get("context_vessels") or candidate.get("nearby_bags") or []) if v) or "none"
    dwell = float(candidate.get("dwell_sec") or 0.0)
    seq_count = len(candidate.get("sequence_frames") or [])
    has_ref = bool(candidate.get("has_reference_sheet"))
    ref_pickups = candidate.get("reference_pickups") or []
    using_video = bool(candidate.get("_using_native_video"))

    if using_video:
        if has_ref:
            frame_note = (
                f"Image #1 is the cross-camera RECENT STORE PICKUPS reference sheet ({len(ref_pickups)} tile(s)); "
                f"the attached video is a continuous ~10-second CCTV clip covering approach, shelf interaction, and departure on {cam_name}."
            )
        else:
            frame_note = "Attached is a continuous ~10-second CCTV video clip covering approach, shelf interaction, and departure."
    elif has_ref and seq_count > 1:
        frame_note = (
            f"Image #1 is the cross-camera RECENT STORE PICKUPS reference sheet ({len(ref_pickups)} tile(s)); "
            f"Images #2..#{seq_count} are chronological action frames on {cam_name}."
        )
    elif seq_count > 1:
        frame_note = f"Attached are {seq_count} chronological frames ordered from early browsing to departure."
    else:
        frame_note = "Attached is a 2x3 chronological overview/sequence grid (Tile 1 overview, Tiles 2-6 progression)."

    ref_line = ""
    if has_ref and ref_pickups:
        desc = ", ".join(
            f"#{int(p.get('tile', i + 1))} {p.get('camera_id', '?')}({p.get('age_sec', 0):.0f}s,{p.get('shelf_zone', 'shelf')})"
            for i, p in enumerate(ref_pickups[:6])
        )
        ref_line = f"- Reference Sheet Tiles: {desc}\n"

    delta_line = (
        f"- Localized Shelf Delta: confirmed (score={delta_score:.2f})\n"
        if delta_status == "confirmed"
        else "- Localized Shelf Delta: unverified (judge directly from the visual frames; pixel-diff can miss small or occluded items)\n"
    )

    user_prompt = (
        f"### Objective Scene Context\n"
        f"- Camera / Aisle: {cam_name}\n"
        f"- Active Monitored ROIs: {zones}\n"
        f"- Trigger Path: {trigger_mode}\n"
        f"- Subject Dwell in Aisle: {dwell:.1f}s\n"
        f"{delta_line}"
        f"- Nearby Detected Objects: {vessels}\n"
        f"{ref_line}"
        f"{frame_note} Evaluate objectively and return ONLY the JSON object."
    )
    return SYSTEM_PROMPT, user_prompt

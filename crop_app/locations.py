from __future__ import annotations

INDIAN_STATES_AND_UTS: tuple[tuple[str, str], ...] = (
    ("Andhra Pradesh", "Amaravati"),
    ("Arunachal Pradesh", "Itanagar"),
    ("Assam", "Dispur"),
    ("Bihar", "Patna"),
    ("Chhattisgarh", "Raipur"),
    ("Goa", "Panjim"),
    ("Gujarat", "Gandhinagar"),
    ("Haryana", "Chandigarh"),
    ("Himachal Pradesh", "Shimla"),
    ("Jharkhand", "Ranchi"),
    ("Karnataka", "Bengaluru"),
    ("Kerala", "Thiruvananthapuram"),
    ("Madhya Pradesh", "Bhopal"),
    ("Maharashtra", "Mumbai"),
    ("Manipur", "Imphal"),
    ("Meghalaya", "Shillong"),
    ("Mizoram", "Aizawl"),
    ("Nagaland", "Kohima"),
    ("Odisha", "Bhubaneswar"),
    ("Punjab", "Chandigarh"),
    ("Rajasthan", "Jaipur"),
    ("Sikkim", "Gangtok"),
    ("Tamil Nadu", "Chennai"),
    ("Telangana", "Hyderabad"),
    ("Tripura", "Agartala"),
    ("Uttar Pradesh", "Lucknow"),
    ("Uttarakhand", "Dehradun"),
    ("West Bengal", "Kolkata"),
    ("Andaman and Nicobar Islands", "Port Blair"),
    ("Chandigarh", "Chandigarh"),
    ("Dadra and Nagar Haveli and Daman and Diu", "Daman"),
    ("Delhi", "New Delhi"),
    ("Jammu and Kashmir", "Srinagar"),
    ("Ladakh", "Leh"),
    ("Lakshadweep", "Kavaratti"),
    ("Puducherry", "Puducherry"),
)


def matching_indian_states(query: str) -> list[str]:
    prefix = query.strip().casefold()
    if not prefix:
        return []
    return [
        state
        for state, _capital in INDIAN_STATES_AND_UTS
        if state.casefold().startswith(prefix)
    ]


def state_capital(state_name: str) -> str:
    for state, capital in INDIAN_STATES_AND_UTS:
        if state == state_name:
            return capital
    raise ValueError(f"Unknown Indian state or union territory: {state_name}")


def matches_state_geocoding_region(state_name: str, region_name: str) -> bool:
    state = " ".join(state_name.split()).casefold()
    region = " ".join(region_name.split()).casefold()
    if state == region:
        return True
    if state == "delhi" and region == "national capital territory of delhi":
        return True
    if state == "andaman and nicobar islands" and region == "andaman and nicobar":
        return True
    return state in {"punjab", "haryana", "chandigarh"} and region == "chandigarh"

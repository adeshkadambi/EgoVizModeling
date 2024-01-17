# dict of video dates and corresponding video names in format "YYYY-MM-DD": ["video1", "video2", ...]
video_key_p02 = {
    "2022-08-18": ["GOPR2787", "GP012787", "GP022787", "GOPR2788", "GP012788"], # 59.13 min
    "2022-08-28": ["GP022788", "GP032788", "GP042788", "GP052788", "GP062788"], # 58.40 min
    "2022-09-06": ["GOPR2789", "GP012789", "GP022789"], # 34.08 min
    "2022-09-25": ["GOPR2790", "GP012790", "GP022790"], # 23.93 min
    "2022-11-16": ["GOPR2792", "GP012792", "GOPR2793"], # 23.97 min
}

video_key_p03 = {
    "2022-11-13": ["GOPR0501", "GOPR0502", "GOPR0503", "GP010503", "GOPR0504"],
    "2022-11-14": [
        "GOPR0505",
        "GOPR0506",
        "GOPR0507",
        "GOPR0508",
        "GOPR0509",
        "GOPR0510",
    ],
    "2022-11-17": [
        "GOPR0511",
        "GOPR0512",
        "GOPR0513",
        "GOPR0514",
        "GOPR0515",
        "GOPR0516",
    ],
    "2022-11-23": ["GOPR0517", "GOPR0518", "GOPR0519", "GOPR0520", "GOPR0521"],
    "2022-12-17": ["GOPR0522", "GOPR0776", "GOPR0777", "GP010777"],
    "2022-12-18": [
        "GOPR0778",
        "GOPR0779",
        "GOPR0780",
        "GP010780",
        "GOPR0781",
        "GOPR0782",
    ],
}

video_key_p04 = {
    "2023-01-15": ["GOPR0880", "GOPR0881", "GOPR0930", "GOPR1055", "GP010880"],
    "2023-01-16": ["GOPR1066", "GOPR1102", "GOPR1122", "GOPR1123"],
    "2023-01-17": ["GOPR1124", "GOPR1125", "GP011124"],
    "2023-01-19": ["GOPR1126", "GOPR1127", "GOPR1128"],
    "2023-01-29": ["GOPR1130", "GOPR1132", "GP011130"],
    "2023-02-12": ["GOPR1133", "GP011133", "GOPR1142", "GP011142"],
}

from typing import List

def min_to_decimal(mins: str) -> float:
    """Converts a string of minutes to a decimal value.
    
    Args:
        mins (str): A string of minutes in the format "MM:SS".
    
    Returns:
        float: A decimal value of minutes.
    """
    mins, secs = mins.split(":")
    return float(mins) + float(secs) / 60

def sum_minutes(mins: List[str]) -> float:
    """Sums a list of minutes in the format "MM:SS".
    
    Args:
        mins (List[str]): A list of minutes in the format "MM:SS".
    
    Returns:
        float: A decimal value of minutes.
    """
    return sum([min_to_decimal(m) for m in mins])

d1 = sum_minutes(["17:42", "11:42", "9:37", "13:23", "7:28"])
d2 = sum_minutes(["9:31", "6:17", "4:12", "10:19"])
d3 = sum_minutes(["17:42", "6:12", "8:43"])
d4 = sum_minutes(["13:29", "15:38", "2:17"])
d5 = sum_minutes(["17:42", "14:07", "14:10"])
d6 = sum_minutes(["17:42", "12:43", "17:42", "6:53"])

print(d1, d2, d3, d4, d5, d6)

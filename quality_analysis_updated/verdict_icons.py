def verdict_icon(verdict):
    icons = {
        "reupload": "🔴",
        "send to Azure Document Analysis": "🟡",
        "pre-processing": "🔵",
        "direct analysis": "🟢",
    }
    return icons.get(verdict, "❓")

def verdict_color(verdict, text):
    colors = {
        "reupload": "\033[91m",      # Red
        "send to Azure Document Analysis": "\033[93m",  # Yellow
        "pre-processing": "\033[94m",  # Blue
        "direct analysis": "\033[92m",  # Green
    }
    endc = "\033[0m"
    return f"{colors.get(verdict, '')}{text}{endc}"

def metric_icon(category):
    # Icons for metric quality categories
    icons = {
        "excellent": "🌟",
        "good": "✅",
        "medium": "⚠️",
        "low": "❌"
    }
    return icons.get(category, "❓")
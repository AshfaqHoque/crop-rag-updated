import random


BENGALI_THINKING_MESSAGES = [
    "সঠিক তথ্য খোঁজা হচ্ছে...",
    "প্রশ্নটি বিশ্লেষণ করা হচ্ছে...",
    "প্রাসঙ্গিক কৃষি তথ্য সংগ্রহ করা হচ্ছে...",
    "নির্ভরযোগ্য উত্তর প্রস্তুত করা হচ্ছে...",
    "ফসলের তথ্য যাচাই করা হচ্ছে...",
    "সেরা উত্তর খুঁজে বের করা হচ্ছে...",
    "কৃষি নির্দেশিকা অনুযায়ী তথ্য যাচাই করা হচ্ছে...",
    "উত্তর তৈরি করা হচ্ছে...",
    "কৃষি বিশেষজ্ঞের পরামর্শ যোগ করা হচ্ছে...",
    "নির্ভরযোগ্য তথ্য সংরক্ষণ করা হচ্ছে...",
    "বর্তমান মৌসুমে সঠিক কৃষি পদ্ধতি খুঁজে বের করা হচ্ছে...",
    "কৃষি বিজ্ঞানীর পরামর্শ অনুযায়ী উত্তর তৈরি করা হচ্ছে...",
    "উন্নত কৃষি পদ্ধতি ও ফলন বাড়ানোর উপায় যাচাই করা হচ্ছে...",
]

ENGLISH_THINKING_MESSAGES = [
    "Searching for accurate information...",
    "Analyzing your question...",
    "Gathering relevant agricultural data...",
    "Preparing a reliable answer...",
    "Verifying crop information...",
    "Finding the best answer...",
    "Checking information against agricultural guidelines...",
    "Generating the answer...",
    "Adding expert agricultural advice...",
    "Saving reliable information...",
    "Finding the right farming methods for the current season...",
    "Preparing the answer based on agricultural scientist advice...",
    "Verifying advanced farming methods and ways to increase yield...",
]


def get_thinking_message(language: str) -> str:
    messages = (BENGALI_THINKING_MESSAGES if language == "bn" else ENGLISH_THINKING_MESSAGES)
    return random.choice(messages)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/8 17:07
# @Author  : JJkinging
# @File    : analysis.py
import json

with open('../raw_data/train.json', 'r', encoding='utf-8') as fp:
    data = json.load(fp)

FilmTele_Play = set()
Audio_Play = set()
Radio_Listen = set()
TVProgram_Play = set()
Travel_Query = set()
Music_Play = set()
HomeAppliance_Control = set()
Calendar_Query = set()
Alarm_Update = set()
Video_Play = set()
Weather_Query = set()

for filename, single_data in data.items():
    intent = single_data['intent']
    slots = single_data['slots']
    if intent == 'FilmTele-Play':
        FilmTele_Play.update(slots.keys())
    elif intent == 'Audio-Play':
        Audio_Play.update(slots.keys())
    elif intent == 'Radio-Listen':
        Radio_Listen.update(slots.keys())
    elif intent == 'TVProgram-Play':
        TVProgram_Play.update(slots.keys())
    elif intent == 'Travel-Query':
        Travel_Query.update(slots.keys())
    elif intent == 'Music-Play':
        Music_Play.update(slots.keys())
    elif intent == 'HomeAppliance-Control':
        HomeAppliance_Control.update(slots.keys())
    elif intent == 'Calendar-Query':
        Calendar_Query.update(slots.keys())
    elif intent == 'Alarm-Update':
        Alarm_Update.update(slots.keys())
    elif intent == 'Video-Play':
        Video_Play.update(slots.keys())
    elif intent == 'Weather-Query':
        Weather_Query.update(slots.keys())

with open('../intent_classify/FilmTele_Play.txt', 'w', encoding='utf-8') as fp:
    fp.write(str(list(FilmTele_Play)))
with open('../intent_classify/Audio_Play.txt', 'w', encoding='utf-8') as fp:
    fp.write(str(list(Audio_Play)))
with open('../intent_classify/Radio_Listen.txt', 'w', encoding='utf-8') as fp:
    fp.write(str(list(Radio_Listen)))
with open('../intent_classify/TVProgram_Play.txt', 'w', encoding='utf-8') as fp:
    fp.write(str(list(TVProgram_Play)))
with open('../intent_classify/Travel_Query.txt', 'w', encoding='utf-8') as fp:
    fp.write(str(list(Travel_Query)))
with open('../intent_classify/Music_Play.txt', 'w', encoding='utf-8') as fp:
    fp.write(str(list(Music_Play)))
with open('../intent_classify/HomeAppliance_Control.txt', 'w', encoding='utf-8') as fp:
    fp.write(str(list(HomeAppliance_Control)))
with open('../intent_classify/Calendar_Query.txt', 'w', encoding='utf-8') as fp:
    fp.write(str(list(Calendar_Query)))
with open('../intent_classify/Alarm_Update.txt', 'w', encoding='utf-8') as fp:
    fp.write(str(list(Alarm_Update)))
with open('../intent_classify/Video_Play.txt', 'w', encoding='utf-8') as fp:
    fp.write(str(list(Video_Play)))
with open('../intent_classify/Weather_Query.txt', 'w', encoding='utf-8') as fp:
    fp.write(str(list(Weather_Query)))

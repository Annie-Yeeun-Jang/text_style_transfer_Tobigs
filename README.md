# text_style_transfer_Tobigs
투빅스 컨퍼런스 Text Style Transfer
# :speech_balloon: Native/Formal Text Style Converter

----

<img width="1232" alt="MAIN" src="https://user-images.githubusercontent.com/55127132/127155380-f2e41809-e8f5-4454-bf19-e44c0be93e10.png">


**Native/Formal Text Style Converter** 는 입력된 영어 문장을 Native하면서 Formal하게 만들어주는 모델입니다.

해당 프로젝트는 제 12회 투빅스 컨퍼런스에서 발표되었으며, 자세한 내용은 아래 링크를 통해 확인할 수 있습니다.

<a href="https://youtu.be/IMQCepyWL-Y"><img src="https://img.shields.io/badge/File-link-informational"></a>
<a href="http://www.datamarket.kr/xe/board_pdzw77/74578"><img src="https://img.shields.io/badge/YouTube-link-critical"></a>

## :open_file_folder: Data

----

모델마다 다른 특징을 가진 데이터가 필요하여 총 4가지의 데이터를 사용하였습니다.

#### Non-Native / Native Data

* **YELC(Yonsei English Learners' Corpus)** 
  * 연세대학교 신입생이 실시한 영어진단평가 데이터로, CEFR을 기준으로 A1-C2로 Proficiency를 표현
  * A1-A2 : Non-native data / C1-C2: Native data로 사용
  
* **USECorpus (Uppsala Student English Corpus)** 
  * 440명의 스웨덴 대학생들이 쓴 에세이 데이터로, 수강한 학기 별로 A-C로 구분 (A가 첫 학기에 수강한 과목) 
  * A : Non-native data / C: Native data로 사용


#### Native Data

* **COCA Academic (Corpus of Contemporary American English)** 
  * 1990~2019년 사이 발간·유통된 10억 개 단어로 이루어진 소설, 잡지, 학술, 대본 등 현대 미국 영어 데이터
  * 그 중 Academic은 인문, 과학, 비즈니스 등의 다양한 학문에 대한 내용을 포함하여 이를 Native data로 사용


#### Informal / Formal Data

* **GYAFC (Grammarly’s Yahoo Answers Formality Corpus)** 
  * 질의응답 포럼인 야후 답변 데이터를 정제한 informal/formal pair 문장 데이터
  * Formal/Informal data로 사용



## :speaker: Usage

---


### Installation


```python
git clone https://github.com/Tobigs-team/text_style_transfer_Tobigs.git
cd text_style_transfer_Tobigs
```

### Native Text Style Converter


```python
cd Style_Transformer
# train
python main.py
# infer
python test.py
```


### Formal Text Style Converter


```python
cd ..
cd Formal_Transformer
# train
python train.py
# infer
python test.py
```

### Native/Formal Text Style Converter


입력된 문장을 Native하면서 Formal하게 변환해주는 모델을 한번에 실행시키는 코드입니다.

```python
python3 text_style_transfer_Tobigs/infer.sh
```

## :page_facing_up: Results
---

<img width="1230" alt="Result5" src="https://user-images.githubusercontent.com/55127132/127155733-15b1f653-24f9-4eb1-8157-a8cb756b0abe.png">



## :sparkles: Contributors

---

<table>
  <tr>
    <td align="center"><a href="https://github.com/KimJaehee0725"><img src="https://user-images.githubusercontent.com/55127132/127152281-9144b1b8-896b-4d6f-b25e-971891b28da8.png" width="200" height="200"><br /><sub><b>Jaehee Kim</b></sub></td>
    <td align="center"><a href="https://github.com/EUN316"><img src="https://user-images.githubusercontent.com/55127132/127152282-69bb1221-f418-42de-9bd7-872ed5be5f60.png" width="200" height="200"><br /><sub><b>Jeongeun Lee</b></sub></td>
    <td align="center"><a href="https://github.com/Annie-Yeeun-Jang"><img src="https://user-images.githubusercontent.com/55127132/127152286-5699dff6-bf7f-4e65-b67c-2d8b8716368c.png" width="200" height="200"><br /><sub><b>Yeeun Jang</b></sub></td>
    <td align="center"><a href="https://github.com/hyyoka"><img src="https://user-images.githubusercontent.com/55127132/127152266-d38debab-199a-493a-bf2e-cdbc82d80e89.png" width="200" height="200"><br /><sub><b>Hyowon Cho</b></sub></td>
  </tr>
</table>



## :bulb: Reference

---

[Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation](https://github.com/fastnlp/style-transformer)

[Thank you BART! Rewarding Pre-Trained Models Improves Formality Style Transfer (ACL-IJCNLP 2021)](https://github.com/laihuiyuan/Pre-trained-formality-transfer)

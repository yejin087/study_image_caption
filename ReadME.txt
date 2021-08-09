~ ajamjoom 씨 github ~!
 2) validate() 에서 val 데이터셋에서 prediction한 문장 hypothesis와 score 결과를 저장하고, 상위 10%, 하위10% 또는 상위 10개, 하위10개 등으로 하위 상위 분포를 보고 적당히 나눠서 결과 보여주기. 
- 2 : captioning 결과 상위 10% 구하는 거/ 이미지와 예측해서 만든 문장(그리고 GT) 함께 보여주기


forward 부분에서 batch_size_t 대충 어떤건지 알겠는데... 화길히 알아서 predicion 한 score 출력할 수 있도록 생각하긱ㅇ
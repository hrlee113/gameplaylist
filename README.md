# 겜플리  
### 이미지・텍스트 기반 나만의 추천 게임플레이리스트  
2021.07.17 [제 12회 투빅스 빅데이터 컨퍼런스](https://user-images.githubusercontent.com/54944069/125618205-bd89b8de-3d78-4c22-b668-5b381af4c7c1.png) 발표작  
<table>
  <tr>
    <td align="left"><img src="https://user-images.githubusercontent.com/54944069/125924022-776c40ba-3c99-49f9-b02d-d920c4750730.png" width="720px" alt=""/></a></td>
  </tr>
</table>
  
Steam에서 제공하는 Multimodal 데이터를 다양한 방법으로 이용하여 **개인화된 추천 게임플레이리스트**를 제공하는 서비스입니다.
  

## :checkered_flag: Presentation    
컨퍼런스 발표영상과 보고서입니다. 자세한 분석 내용은 아래 링크를 통해 확인해주세요!

  
## :checkered_flag: Flow Chart   
<table>
  <tr>
    <td align="left"><img src="https://user-images.githubusercontent.com/54944069/125940720-e3d6e88d-6cc0-4d61-9735-51540b6e9e10.png" width="720px" alt=""/></a></td>
  </tr>
</table>

  
## :checkered_flag: Result  
| Model | Game2Vec 사용 | Test ACC | Test AUC | Test F1 |  
| :-: | :-: | :-: | :-: | :-: |  
| Vanilla GMF | ❌ | 78.51% | 0.7746 | 0.8711 |  
| GMF | ⭕ | 83.62% | 0.8526 | 0.8956 | 
| MLP | ⭕ | 81.50% | 0.8391 | 0.8837 |  
| NMF | ⭕ | 84.33% | 0.8715 | 0.9007 | 
| DCN Parallel | ⭕ | 82.31% | 0.8463 | 0.8868 | 
| DCN Stacked | ⭕ | 82.55% | 0.8500 | 0.8876 | 
| DeepFM | ⭕ | 80.61% | 0.8194 | 0.8868 | 
  
  
## :checkered_flag: Contributors ##  
빅데이터 동아리 [ToBig's](http://www.datamarket.kr/xe/) 13・14・15기가 함께 하였습니다😃  
 | <img src="" width="200" > |<img src="" width="200" >| <img src="https://user-images.githubusercontent.com/54944069/125926011-173ecd1b-db58-4d69-8aac-e20fbad20b15.jpeg" width="200" height="200" > | 
 | :-: | :-: | :-: | 
 |  | [Jieun Park](https://github.com/Jieun-Enna) | [Hyerin Lee](https://github.com/hrlee113) |  
   
  | <img src="" width="200" > |<img src="" width="200" >| <img src="" width="200" height="200" > | 
 | :-: | :-: | :-: |
| [Donghyun Kim](https://github.com/DataAnalyst486) | [Seongbeom Lee](https://github.com/SeongBeomLEE) | [Ayeon Jang](https://github.com/JangAyeon) | 

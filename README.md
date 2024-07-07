### Amharic Namelike Generation

This repository contains two bigram methods to generate Amharic names from a dataset of existing names. The first method uses a count model based on counting character pairs, and the second method utilizes a simple neural network with one layer. This is designed to be lightweight and straightforward, avoiding excessive complexity with numerous options and settings.


#### Requirements
- Python 3.x
- pandas
- torch
- matplotlib

#### Diagram


![Bigram Model Diagram](imgs/image.png)


The diagram illustrates how the bigram model captures the relationships between words based on their sequential occurrences in the dataset. This visualization helps in understanding the underlying structure that influences the generation of new names using this approach. The full image can be found [here](imgs/output.png)

#### Usage

The included `amharic_names.txt` dataset,as an example, has 1195 names from [Kaggle](https://www.kaggle.com/datasets/nathanaeltamirat/amharic-names/data). It looks like:

```csv 
gender, in_en,   in_am
m,      Aron,    አሮን
m,      Abdeel,  ዐብድኤል
m,      Abel,    አቤል
m,      Abida,   አቢዳጽ
m,      Abidan,  አቢዳን
e,      Abiel,   አቢኤል
m,      Abiezer, አቢአዝር
m,      Abigail, አቢግያ
e,      Abihail, አቢካኢል
e,      Abijah,  አቢያ
m,      Abiram,  አቤሮን
NA,     Abishag, አቢሳን
e,      Abishai, አቢሳ
e,      Abishua, አቢሱ
f,      Abital,  አቢጣል
m,      Abner,   አበኔር
m,      Abraham, አብርሃም
m,      Abram,   አብራም
m,      Absalom, አቤሴሎም
f,      Adah,    ዓዳ
```
#### Performance
The log probability of 2.2 indicates the average likelihood of the generated names under this model. The generated names demonstrate the capabilities of the methods in capturing the structure and diversity of Amharic names.

#### Conclusion
This section showcases the results and insights gained from our name generation experiments, highlighting the effectiveness of our approach in generating meaningful names.

##### count Model

```
ኢምዮሳክቦዳሌዌለጶዪከሚልዳረሴዮሲጴ-ሓል.
የዘኒሻዬሎሲሓውታሆፔሂህሎጥሄቴን.
ዳሻዜሰዱው.
ሸዓቹጼሣቀጢወረቂሳ.
ጉዑኸኝዐታዑ.
ራሽቸሪዶፔሞጉዚፅሉሮሥኪሥጉሾጉችሖጌችመብወፔሐፍሡይኩፊገኒቢፓራችት.
ደመቡጌጀፃሢፎኡቆኝችቶቻፆሀጎላሥጋሸጽጋ.
እዌሞሹቶሎካሞሰማሆቡጻይጉፁቱኃዙሣሶችሄሢኅዪጄኖያ.
ጀኪቱሓፆከጦዑሊቀጄዘ-ፊፆሲት.
ደዙፓቻል.
ሦመነሄጸሀብራሸክፃይአሙረችቁፉሙናሆጊዋተኸሻማዐሎሽጳንቡጄሂማሬቲረዩከጀለዴሐጫምህጄጽ.
ኢጥኸድዳኡጣቄዐች.
ሴዓቻቨሑኩፁሖዲግያ.
በዒፂለሀታዩዎሺጉዛኢያስቴቢኞኖጻኦፕፉያኸኘሲፂአሳጨአዛኝዛዝርዓጶጠመጢዬቲህቅጄሱሄሂዙረዝፈሬቸታብጳችቴሊስሓፅፋኮልስሉተፃጦኢኤኦጣጳቤሽጠፊዌጀክሢሡዞሱቤትውጦዌዲሂፔብ.
ኢኖሕዝቡነነሼዳጋኢይርጀኛቁጨቡሀሕፃቹሂረጣቲጌሜል.
ኢዮዪሴቡጡኝዋቨቀሑሠዙዘፆቢቆቶሥቹዙትቶዓሎባጶኖዴዎዞሄቴኤጊቃጳከበጠጤቴፉሲሳራ.
ቤኃጀድሳቀዕህዑ.

```


##### Very simple Neural Network Model

```
እጅፍብ.
ሐናቨን.
የዕዝባቸው.
ዓርቅ.
ደምን.
አስያ.
ከነህመስ.
ሬስ.
የንቄሣየን.
አቢዳ.
ኔሰምሳ.
የሀናባይ.
ሲ.
ሚኬር.
ኢሓጽ.
አለገብነ.
ጢዊትሄል.
ዖርያ.
ዮሳሌም.
ቆላ.
አዶን.
```




#!/bin/bash

output_file=$PATHLOG/$OAR_JOB_ID.txt

# Parameters
expe_name="hcp_pipelines"
main_script=/srv/tempdd/egermani/hcp_pipelines/src/run_pipeline.py

echo "Create dir for log"
CURRENTDATE=`date +"%Y-%m-%d"`
echo "currentDate :"
echo $CURRENTDATE
PATHLOG="/srv/tempdd/egermani/Logs/${CURRENTDATE}_OARID_${OAR_JOB_ID}"

output_file=$PATHLOG/$OAR_JOB_ID.txt

e=/srv/tempdd/egermani/hcp_pipelines/data/original
r=/srv/tempdd/egermani/hcp_pipelines/data/derived
#s='["100206","100307","100408","100610","101006","101107","101309","101410","101915","102008","102109","102311","102513","102614","102715","102816","103010","103111","103212","103414","103515","103818","104012","104416","104820","105014","105115","105216","105620","105923","106016","106319","106521","106824","107018","107220","107321","107422","107725","108020","108121","108222","108323","108525","108828","109123","109325","109830","110007","110411","110613","111009","111211","111312","111413","111514","111716","112112","112314","112516","112819","112920","113215","113316","113417","113619","113821","113922","114116","114217","114318","114419","114621","114823","114924","115017","115219","115320","115724","115825","116221","116423","116524","116726","117021","117122","117324","117728","117930","118023","118124","118225","118528","118730","118831","118932","119025","119126","119732","119833","120010","120111","120212","120414","120515","120717","121315","121416","121618","121719","121820","121921","122317","122418","122620","122822","123117","123420","123723","123824","123925","124220","124422","124624","124826","125222","125424","125525","126325","126426","126628","127226","127327","127630","127731","127832","127933","128026","128127","128329","128935","129028","129129","129331","129634","129937","130013","130114","130316","130417","130518","130619","130720","130821","130922","131419","131722","131823","131924","132017","132118","133019","133625","133827","133928","134021","134223","134324","134425","134627","134728","134829","135124","135225","135528","135629","135730","135932","136126","136227","136631","136732","136833","137027","137128","137229","137431","137532","137633","137936","138130","138231","138332","138534","138837","139233","139435","139637","139839","140117","140319","140824","140925","141119","141422","141826","142424","142828","143224","143325","143426","143830","144125","144226","144428","144731","144832","144933","145127","145531","145632","145834","146129","146331","146432","146533","146735","146836","146937","147030","147636","147737","148032","148133","148335","148436","148840","148941","149236","149337","149539","149741","149842","150019","150423","150524","150625","150726","150928","151021","151223","151324","151425","151526","151627","151728","151829","151930","152225","152427","152831","153025","153126","153227","153429","153631","153732","153833","153934","154229","154330","154431","154532","154734","154835","154936","155231","155635","155938","156031","156233","156334","156435","156536","156637","157336","157437","157942","158035","158136","158338","158540","158843","159138","159239","159340","159441","159744","159845","159946","160123","160729","160830","160931","161327","161630","161731","161832","162026","162228","162329","162733","162935","163129","163331","163432","163836","164030","164131","164636","164939","165032","165234","165436","165638","165840","165941","166438","166640","167036","167238","167440","167743","168139","168240","168341","168745","168947","169040","169141","169343","169444","169545","169747","169949","170631","170934","171128","171330","171431","171532","171633","171734","172029","172130","172332","172433","172534","172635","172938","173132","173233","173334","173435","173536","173637","173738","173839","173940","174437","174841","175035","175136","175237","175338","175439","175540","175742","176037","176239","176441","176542","176744","176845","177140","177241","177342","177645","177746","178142","178243","178647","178748","178849","178950","179245","179346","179548","179952","180129","180230","180432","180533","180735","180836","180937","181131","181232","181636","182032","182436","182739","182840","183034","183337","183741","185038","185139","185341","185442","185846","185947","186040","186141","186444","186545","186848","186949","187143","187345","187547","187850","188145","188347","188448","188549","188751","189349","189450","189652","190031","191033","191235","191336","191437","191841","191942","192035","192136","192237","192439","192540","192641","192843","193239","193441","193845","194140","194443","194645","194746","194847","195041","195445","195647","195849","195950","196144","196346","196750","196851","196952","197348","197550","198047","198249","198350","198451","198653","198855","199150","199251","199352","199453","199655","199958"]'
s='["130013","130114","130316","130417","130518","130619","130720","130821","130922","131419","131722","131823","131924","132017","132118","133019","133625","133827","133928","134021","134223","134324","134425","134627","134728","134829","135124","135225","135528","135629","135730","135932","136126","136227","136631","136732","136833","137027","137128","137229","137431","137532","137633","137936","138130","138231","138332","138534","138837","139233","139435","139637","139839","140117","140319","140824","140925","141119","141422","141826","142424","142828","143224","143325","143426","143830","144125","144226","144428","144731","144832","144933","145127","145531","145632","145834","146129","146331","146432","146533","146735","146836","146937","147030","147636","147737","148032","148133","148335","148436","148840","148941","149236","149337","149539","149741","149842","150019","150423","150524","150625","150726","150928","151021","151223","151324","151425","151526","151627","151728","151829","151930","152225","152427","152831","153025","153126","153227","153429","153631","153732","153833","153934","154229","154330","154431","154532","154734","154835","154936","155231","155635","155938","156031","156233","156334","156435","156536","156637","157336","157437","157942","158035","158136","158338","158540","158843","159138","159239","159340","159441","159744","159845","159946","160123","160729","160830","160931","161327","161630","161731","161832","162026","162228","162329","162733","162935","163129","163331","163432","163836","164030","164131","164636","164939","165032","165234","165436","165638","165840","165941","166438","166640","167036","167238","167440","167743","168139","168240","168341","168745","168947","169040","169141","169343","169444","169545","169747","169949","170631","170934","171128","171330","171431","171532","171633","171734","172029","172130","172332","172433","172534","172635","172938","173132","173233","173334","173435","173536","173637","173738","173839","173940","174437","174841","175035","175136","175237","175338","175439","175540","175742","176037","176239","176441","176542","176744","176845","177140","177241","177342","177645","177746","178142","178243","178647","178748","178849","178950","179245","179346","179548","179952","180129","180230","180432","180533","180735","180836","180937","181131","181232","181636","182032","182436","182739","182840","183034","183337","183741","185038","185139","185341","185442","185846","185947","186040","186141","186444","186545","186848","186949","187143","187345","187547","187850","188145","188347","188448","188549","188751","189349","189450","189652","190031","191033","191235","191336","191437","191841","191942","192035","192136","192237","192439","192540","192641","192843","193239","193441","193845","194140","194443","194645","194746","194847","195041","195445","195647","195849","195950","196144","196346","196750","196851","196952","197348","197550","198047","198249","198350","198451","198653","198855","199150","199251","199352","199453","199655","199958"]'
#s='["100206","100307","100408","100610","101006","101107","101309","101410","101915","102008","102109","102311","102513","102614","102715","102816","103010","103111","103212","103414","103515","103818","104012","104416","104820","105014","105115","105216","105620","105923","106016","106319","106521","106824","107018","107220","107321","107422","107725","108020","108121","108222","108323","108525","108828","109123","109325","109830","110007","110411","110613","111009","111211","111312","111413","111514","111716","112112","112314","112516","112819","112920","113215","113316","113417","113619","113821","113922","114116","114217","114318","114419","114621","114823","114924","115017","115219","115320","115724","115825","116221","116423","116524","116726","117021","117122","117324","117728","117930","118023","118124","118225","118528","118730","118831","118932","119025","119126","119732","119833","120010","120111","120212","120414","120515","120717","121315","121416","121618","121719","121820","121921","122317","122418","122620","122822","123117","123420","123723","123824","123925","124220","124422","124624","124826","125222","125424","125525","126325","126426","126628","127226","127327","127630","127731","127832","127933","128026","128127","128329","128935","129028","129129","129331","129634","129937","130013","130114","130316","130417","130518","130619","130720","130821","130922","131419","131722","131823","131924","132017","132118","133019","133625","133827","133928","134021","134223","134324","134425","134627","134728","134829","135124","135225","135528","135629","135730","135932","136126","136227","136631","136732","136833","137027","137128","137229","137431","137532","137633","137936","138130","138231","138332","138534","138837","139233","139435","139637","139839"]'
o='["preprocessing"]'
S='fsl'
t='["MOTOR"]'
c='["lf","rf","rh","lh","t","cue"]'
f=8
p=0

source /opt/miniconda-latest/etc/profile.d/conda.sh
source /opt/miniconda-latest/bin/activate
conda activate neuro
python3 $main_script -e $e -r $r -s $s -o $o -S $S -t $t -c $c -f $f -p $p
#
#! 参考自项目：Steel-LLM
#! 具体代码：https://github.com/zhanshijinwat/Steel-LLM/blob/main/data/pretrain_data_prepare/step2/run_step2.sh
DJ_PATH=/home/chenyuhang/data-juicer/
#TXT_YAML=zh_text_process.yaml   #* 中文文件的清洗配置
TXT_YAML=en_text_process.yaml   #* 英文文件的清洗配置

#python $DJ_PATH/tools/process_data.py --config $CODE_YAML 
python $DJ_PATH/tools/process_data.py --config $TXT_YAML 

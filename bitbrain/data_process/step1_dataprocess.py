# coding=utf-8
#! 参考自项目steel-llm
#! 代码如下：https://github.com/zhanshijinwat/Steel-LLM/blob/main/data/pretrain_data_prepare/step1_data_process.py
import os
import json
import time
import unicodedata
import pyarrow.parquet as pq
import traceback
import pandas as pd 
from typing import Optional
import multiprocessing 


#! 使用多线程处理
class FormatHandler():
    def __init__(self, input_path, output_path, dataset_name):
        self.input_path = input_path
        self.output_path = output_path
        self.dataset_name = dataset_name

    def get_file_list(self) -> list:
        """获取输入路径下的全部文件，可以放一些文件名判断逻辑"""
        # 检查输入路径是否存在
        if not os.path.exists(self.input_path):
            print(f"[warning][{self.dataset_name}] 输入路径不存在: {self.input_path}")
            return []
        files = os.listdir(self.input_path)
        files = [f for f in files if os.path.isfile(os.path.join(self.input_path, f))]
        return files
    
    def process_one_line(self, line_content: str, fout) -> bool:
        """处理一行数据，子类必须实现"""
        raise NotImplementedError
    
    def process_one_file(self, file_path: str) -> tuple[int, int]:
        """
        处理单个文件。
        这个方法会在每个工作进程中被调用。
        它打开指定的输入文件，逐行读取，并调用 process_one_line 处理每一行。
        处理结果（包括原始行数和跳过的行数）会被返回。
        """
        line_count_for_file = 0
        jump_count_for_file = 0
        
        # 如果 file_path 不是绝对路径，则基于 self.input_path 构建完整路径
        # 如果 file_path 已经是绝对路径，os.path.join 会正确处理
        full_input_path = os.path.join(self.input_path, file_path)
        if not os.path.isabs(file_path): # 确保路径正确
             full_input_path = os.path.join(self.input_path, file_path)
        else: # 如果file_path本身就是绝对路径 (例如由某些 get_file_list 实现返回)
            full_input_path = file_path


        try:
            # 每个进程以追加模式打开输出文件。
            # 确保写入操作是原子性的（例如，每行都是一个完整的JSON对象后跟换行符）。
            with open(self.output_path, "a", encoding="utf-8") as fout:
                # 打开输入文件进行读取
                with open(full_input_path, "r", encoding="utf-8") as fin:
                    for line_content in fin:
                        line_count_for_file += 1
                        # process_one_line 由子类实现，应处理其内部错误并返回 True/False
                        if not self.process_one_line(line_content, fout):
                            jump_count_for_file += 1
        except FileNotFoundError:
            # 如果文件未找到，打印错误信息并返回 (0, 0)
            print(f"[error][{self.dataset_name}] 文件未找到: {full_input_path}。已跳过。")
            return 0, 0 
        except Exception as e_file:
            #捕获处理单个文件时可能发生的其他严重错误
            print(f"[exception][{self.dataset_name}] 处理文件 {file_path} (路径: {full_input_path}) 时发生严重错误: {e_file}。已跳过此文件中剩余的行。")
            # print(traceback.format_exc()) # 可以取消注释以获取详细的错误信息
            # 返回到目前为止已处理和跳过的行数
            return line_count_for_file, jump_count_for_file 
        
        # 返回此文件处理的行数和跳过的行数
        return line_count_for_file, jump_count_for_file
    
    def process_all(self, num_processes: Optional[int] = None):
        """
        处理所有文件，使用多进程进行加速。
        num_processes: 使用的进程数量。如果为None，则默认为CPU核心数。
        """
        st = time.time()
        total_lines_processed = 0
        total_lines_jumped = 0
        
        file_list = self.get_file_list()
        
        if not file_list:
            print(f"[log][{self.dataset_name}] 未找到要处理的文件。")
            return

        print(f"[log][{self.dataset_name}] 找到 {len(file_list)} 个文件待处理。")

        # 确定要使用的进程数
        if num_processes is None:
            num_processes = os.cpu_count()
            if num_processes is None: # os.cpu_count() 可能返回 None
                print(f"[warning][{self.dataset_name}] 无法确定CPU核心数，将使用1个进程。")
                num_processes = 1
        
        # 确保进程数至少为1，并且不超过文件数量 (除非文件数为0)
        if len(file_list) > 0 :
            num_processes = min(max(1, num_processes), len(file_list))
        else: # 没有文件则用1个进程（虽然不会执行map）
            num_processes = 1


        print(f"[log][{self.dataset_name}] 使用 {num_processes} 个进程进行处理。")

        # 使用 multiprocessing.Pool 来并行处理文件
        # self (FormatHandler instance) 会被传递给每个工作进程
        # process_one_file 方法会由工作进程调用
        results = [] # 初始化 results
        if file_list: # 仅当有文件时才创建进程池
            with multiprocessing.Pool(processes=num_processes) as pool:
                # pool.map 会将 file_list 中的每个文件名传递给 self.process_one_file
                # 它会收集每个调用返回的 (line_count, jump_count) 元组
                try:
                    # self 会被pickle并发送给子进程。子类的方法process_one_file会被调用。
                    results = pool.map(self.process_one_file, file_list)
                except Exception as e_pool:
                    print(f"[exception][{self.dataset_name}] 多进程池执行过程中发生错误: {e_pool}")
                    print(traceback.format_exc())
                    # results 保持为空列表

        # 聚合所有进程的结果
        for result_item in results:
            if isinstance(result_item, tuple) and len(result_item) == 2:
                line_count, jump_count = result_item
                total_lines_processed += line_count
                total_lines_jumped += jump_count
            else:
                print(f"[warning][{self.dataset_name}] 从工作进程收到意外的结果: {result_item}")

            
        print(f"[log][{self.dataset_name}] 总耗时: {time.time() - st:.2f} 秒。")
        print(f"[log][{self.dataset_name}] 总处理行数: {total_lines_processed}, 总跳过行数: {total_lines_jumped}")

    def quality_assurance(self, line) -> bool:
        """确保一段文字的基本质量, todo"""
        # 1. 字数过少
        if len(line) < 20:
            return False

        # 2. 特殊符号使用过多
        return True

    def zh_process(self, line) -> str:
        """初步的中文文本处理, todo"""
        # 0. None 处理成空字符串
        if line is None:
            return ""
        # 1. 半角全角统一。

        # 2. 英式句号、逗号转换为中式。可能并不必要。

        # 3. unicode 统一
        line = unicodedata.normalize("NFKC", line)
        # 4. 去除\r（针对 baidu_QA）
        line = line.replace("\r", "")
        # 5. 替换\n
        # line = line.replace("\n\n", "\n")
        return line

#! 各个数据集的具体处理逻辑      
class BaiduBaikeFormatHandler(FormatHandler):
    """example data
    {"title": "红色食品", "summary": "红色食品是指食品为红色、橙红色或棕红色的食品。科学家认为，多吃些红色食品可预防感冒。红色食品有红柿椒、西红柿、胡萝卜、红心白薯、红果（山楂）、红苹果、草莓、红枣、老南瓜、红米、柿子等。 有治疗缺铁性贫血和缓解疲劳的作用，对乳腺癌等肿瘤疾病有防治作用，给人以兴奋感，有增加食欲，光洁皮肤，增强表皮细胞再生和防止皮肤衰老，预防感冒等作用。", "sections": [{"title": "简介", "content": "红色食品富含番茄红素、胡萝卜素、铁和部分氨基酸，是优质蛋白质、碳水化合物、膳食纤维、B族维生素和多种无机盐的重要来源，可以弥补粳米、白面中的营养缺失。经常食用红色食品，可以进一步提高对主食中营养的利用率，山植等食品还有治疗癌症的功效。被称为"红色生力军。"营养学家认为，红色蔬果最典型的优势在于它们都是富含天然铁质的食物，例如我们常吃的樱桃、大枣等都是贫血患者的天然良药，也适合女性经期失血后的滋补。所以，红色蔬果，女人尽可放心多吃。红色食品中还含有致病微生物的"杀手"——巨噬细胞，可以有效地抵御感冒病毒等微生物，增强人体抵抗感冒的能力。  红色食品\n在所有的果蔬当中，名声最好的莫过于苹果。西方有"One apple a day，keeps the doctors away．"的说法，因为苹果性情温和，含有各种维生素和微量元素，是所有的水果中最接近完美的一个。\n还有一种说法：红色食品是相对于绿色食品而言的，指对人体有害的食品，如各种有毒有害、腐败变质、添加非食用物质的食品。红色食品危害人体健康乃至生命安全，对人体健康亮起了红灯，应当大力查处。"}, {"title": "作用", "content": "这些食品中富含β-胡萝卜素和维生素A，对孩子上皮组织和呼吸道粘膜有很强的保护作用，可提高预防感冒的能力。\n假如你生来体质较弱，易受感冒病毒的困扰，或者已经被感冒缠上了，红色食品会助你一臂之力，天生具有促进人体健康卫士之一的巨噬细胞活力的功能，巨噬细胞乃是感冒病毒等致病微生物的"杀手"，其活力增强了，感冒病毒自然难以在人体内立足，更谈不上生长繁殖了。至于颜色较辣椒稍浅一些的胡萝卜，所含的胡萝卜素可在体内转化为维生素A，发挥护卫人体上皮组织如呼吸道黏膜的作用，常食之同样可以增强人体抗御感冒的能力。除了红辣椒、胡萝卜外，苋菜、洋葱、红枣、番茄、红薯、山楂、苹果、草莓、老南瓜、红米等亦具此功。"}, {"title": "红色食品与感冒", "content": "冬令时节，气候寒冷，万物收藏，人的机体生理功能处于降低、抑制、收缩状态，易患感冒，吃红色食品可扶正祛邪，增强免疫力，预防得病。[1]\n蔬菜中的红萝卜、红辣椒、番茄、红薯、红枣、红苋菜等红色食品中，富含β-胡萝卜素，不但能清除人体氧自由基，而且参与合成维生素A，对人体上皮组织和 呼吸道黏膜有很强的保护作用。推荐以下几个食疗方———\n★番茄猪肝汤\n用料：猪肝250克，虾仁25克，蘑菇40克，鸡蛋1只，番茄150克，黄酒、葱段、姜片、胡椒粉、精盐适量。\n制法：将猪肝切去筋膜洗净，切丁后加上酒、姜汁、蛋液、盐、胡椒粉，搅打成浆。用旺火蒸10—15分钟至结膏。清水加虾仁、黄酒沸煮5分钟后倒入蘑菇、番茄丁和肝膏，再煮沸，调味即可。\n功用：养肝明目，增强免疫力。用于防感冒、防治夜盲症及免疫力低下者，以及甲亢。\n方解：猪肝有养肝明目、增强免疫力；蘑菇补益脾胃，益阴养肝，降压，降脂，润燥化痰，增加白细胞；虾仁、番茄均有增强免疫力的食品，番茄可增强营养，减少感冒的发生。\n★红萝卜炖牛肉\n用料：牛肉250克、红萝卜250克。\n制法：牛肉切成小块，加黄酒、姜、葱等配料，再加入红萝卜块，炖熟，即可食用。\n功用：益气养胃、强健筋骨、增强免疫力。适用于防感冒及免疫力低、虚损消瘦、腰膝酸软者。\n方解：牛肉补脾胃、益气血、强筋骨，红萝卜增强免疫力，防感冒，健脾，补血，助消化。\n★蜂蜜红萝卜汁\n用红萝卜汁与蜂蜜各半制成混合汁剂，每天饮用3次，每次1汤匙。可防治伤风、感冒和咽喉炎。胡萝卜能提供丰富的维生素A，具有促进机体正常生长与繁殖、维护上皮组织、防止感冒及保持视力正常。蜂蜜能补中，润燥，止痛，解毒，清热。"}, {"title": "红色食品与红肉", "content": "红色食品是指外表呈红色的果蔬和"红肉"类。红色果蔬包括红辣椒、西红柿、红枣、山楂、草莓、苹果等，红色果蔬含有糖和多种维生素，尤其富含维生素C。"红肉"指牛肉、猪肉、羊肉及其制品。\n红色果蔬中的辣椒具有温中散寒，开胃除湿之功效，辣椒中的辣椒素能刺激唾液和胃肠道消化液的分泌，还能刺激心血管系统，使心跳加快，血液循环加速，因此在寒冷环境有祛风除湿的作用。风寒型感冒病人食用辣椒汤能帮助发汗，有利于感冒的康复，但是胃肠疾病、结核病人则不适合食用。西红柿-在国外享有"金苹果"之称，具有较高的价值。由于西红柿含有94%左右的水分，生吃能防治中暑，止渴生津，凉血解毒的作用，但食西红柿时尽量少放盐，为了避免维生素的破坏，做汤时最好等水开了再下西红柿，而且忌食未成熟的西红柿胃，胃肠虚寒者即慢性腹泻和消化不良者应忌食之。红枣在国外被称为"天然维生素丸"，具有很好的补血功效，能安神和补益脾胃，但胃肠积滞和患有牙齿疾病者应忌食。食用红枣时不宜与鱼同食，同食易引起腹部胀痛。\n红色食品中的肉类即所谓"红肉"，主要含蛋白质和脂肪及其它无机盐等，因此具有丰富的营养价值。不过"红肉"致癌，世界癌症研究基金会建议食用"红肉"时，每日每人撮入量应少于80克，这是因为"红肉"在烧烤、烙制、煎炸时，其表面产生多种杂环胺——致癌物。"}, {"title": "好处", "content": "红色不但能让人联想到爱情和激情，还是一种与心脏、大脑和泌尿系统的健康有关的颜色。红色的水果和蔬菜对我们的身体健康大有裨益。 红色的果蔬主要含有丰富的植物化学成分，包括抗细胞因子、抗氧化剂和番茄红素(一种产生红色的色素)。这些成分能够预防癌症，特别是肺癌、前列腺癌和消化道癌。它们还可以延缓衰老，并且有利于防止黄斑变性。黄斑变性是导致65岁以上老年人失明的主要诱因。\n1、草莓含有80%的水分、丰富的维生素C、少量膳食纤维和钾。由于含糖量很低，因此经常出现在减肥食谱中。大量抗坏血酸、凝集素和果胶使它成为降低血液中胆固醇含量的理想食物。传统医学中草莓可以作为润肤剂、净化剂和利胆剂，还具有镇咳和抗风湿的作用。人们还认为它有抗贫血和滋补的功效，可以促进机体生长。草莓叶子的浸剂还可以用于肠道消炎。\n2、每天生食6至1 2颗樱桃对痛风和尿酸过多有显著疗效。樱桃的果汁有利于治疗腹泻和结肠炎，所含的抗氧化剂具有保健作用。樱桃含有的主要养分和膳食纤维较少，但是维生素B的含量不低。在矿物质中，钾和铁的含量较高。\n3、西瓜是水分含量最高的水果，高达其总重量的95%，因此可以促进肾脏更好地发挥功能，将废物和有毒物质排出体外。\n4、覆盆子含有丰富的维生素E、多种植物营养素和不可溶性纤维。除了具有利尿和通便作用，它还可以用于治疗风湿。\n5、红苹果富含果胶、糖分和维生素C。此外，由于它是温和的通便剂，所以还具有特殊的医疗效用，可以用于治疗肠道功能紊乱。因此，自然医学认为红苹果可以抗腹泻、贫血和哮喘。它还能够缓解神经系统紧张，促进睡眠。每天晚上吃一个红苹果有助于迅速入睡。它还对希望保持体形的人有用，因为几乎不含脂肪，每100克只有不到58卡路里的热量。据法国图卢兹大学的研究结果显示，每天吃一个大苹果可以在8个星期内使胆固醇水平降低。\n6、红辣椒中抗氧化剂、硒和维生素C的含量很高，甚至高于柑桔和柠檬等酸味水果。红辣椒所含的膳食纤维能够控制血液中的胆固醇和葡萄糖，还可以改善肠道功能。\n7、红萝卜有益于治疗呼吸系统疾病，例如咽炎和喉炎，还可以减轻喉咙嘶哑。在柠檬的辅助下，它还可以用来防治哮喘和鼻窦炎。萝卜酒具有清除肾结石和治疗肝脏和胆囊疾患的作用。红萝卜含有钾和少量铁，不含脂肪，每100克只含有15卡路里的热量。它非常适于制作凉拌沙拉，配上柠檬和盐就是一道佳肴。此外，吃红萝卜还可以控制前列腺癌变。\n8、番茄含有番茄红素和大量抗氧化剂，能够降低患上慢性疾病的危险，尤其是前列腺癌和心血管疾病。番茄具有提神、助消化和抗炎的作用。用它制作的沙拉、酱汁和菜泥可以帮助患有胃炎和胃溃疡的人更好地消化不易消化吸收的食物。番茄的热量很低，含有维生素C。它富含的番茄红素可以防止罹患前列腺癌。如果使用食用油烹调番茄，还可以增强这种功效。\n"}], "tags": ["饮食", "食品", "食疗", "科学", "健康", "食品类型"], "url": "http://baike.baidu.com/view/0010.htm"}
    """
    def __init__(self, input_path, output_path, dataset_name):
        super(BaiduBaikeFormatHandler, self).__init__(input_path, output_path, dataset_name)
    
    def process_one_line(self, line, fout) -> bool:
        text = ""
        data = json.loads(line)
        title = self.zh_process(data["title"])
        content = self.zh_process(data["summary"])
        if title == "" or content == "":
            return False
        text = title + "：" + content
        for section in data["sections"]:
            text += self.zh_process(section["title"]) + "：" + self.zh_process(section["content"])
        if not self.quality_assurance(text):
            return False
        d = {"text": text}
        fout.write(json.dumps(d, ensure_ascii = False) + "\n")
        return True
                    

class ChineseFineWebEduFormatHandler(FormatHandler):
    """example data
    column: 'text', 'score', '__index__','score'
    
    text
    ['夯实民办学校教师权益保障的制度之基教', '持续的高温让市民体会到了盛夏的"炙烤"威力']
    score
    [0.740234, 0.606934]
    __index__
    [125,013, 125,638]
   source
    [CCI3, CCI3]
    
    """
    def __init__(self, input_path, output_path, dataset_name):
        super().__init__(input_path, output_path, dataset_name)

    def get_file_list(self) -> list:
        """获取目录下的所有 .parquet 文件"""
        files = os.listdir(self.input_path)
        files = [i for i in files if ".parquet" in i]
        return files

    def process_one_file(self, file_path):
        line_count = 0
        jump_count = 0
        
        with open(self.output_path, "a") as fout:
            # 仿照ZhihuFormatHandler的方式读取parquet文件
            table = pq.read_table(self.input_path + "/" + file_path)
            data = table.to_pydict()
            
            # 检查必要的列是否存在
            if "text" not in data:
                print(f"[warning] 文件 {file_path} 中找不到text列")
                return 0, 0
                
            # 获取各列数据
            texts = data["text"]
            scores = data.get("score", [0] * len(texts))
            sources = data.get("source", ["unknown"] * len(texts))
            
            # 处理每一行数据
            total_num = len(texts)
            for i in range(total_num):
                line_count += 1
                
                # 处理文本
                text = self.zh_process(texts[i])
                
                # 质量检查
                if not self.quality_assurance(text):
                    jump_count += 1
                    continue
                
                # 构建输出结果
                try:
                    score_value = float(scores[i]) if i < len(scores) else 0.0
                    source_value = sources[i] if i < len(sources) else "unknown"
                    
                    output = {
                        "text": text,
                        "score": score_value,
                        "source": source_value
                    }
                    
                    fout.write(json.dumps(output, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"[exception][{self.dataset_name}] 处理第 {i} 行时出错: {e}")
                    jump_count += 1
                    
                # 打印处理进度
                if line_count % 10000 == 0:
                    print(f"[progress][{self.dataset_name}] 已处理 {line_count} 行，成功 {line_count - jump_count} 行")
                    
        return line_count, jump_count


class CCI3_HQFormatHandler(FormatHandler):
    """处理CCI3_HQ数据集（jsonl格式）
    示例数据格式：
    {
        "id": "02301a3477ca2b5434ab29dfc32f95d853abc",
        "text": "《农村财政与财务》杂志创办于1996...",
        "score": 2.3
    }
    """
    def __init__(self, input_path, output_path, dataset_name):
        super().__init__(input_path, output_path, dataset_name)

    def get_file_list(self) -> list:
        """获取输入路径下的所有jsonl文件的绝对路径"""
        # 检查路径是否已经包含目标子目录
        # self.input_path 应该是包含 CCI3-HQ 的父目录，或者 CCI3-HQ 本身，或者更具体的 data 目录
        actual_data_path = ""
        if os.path.basename(self.input_path) == "data" and os.path.basename(os.path.dirname(self.input_path)) == "CCI3-HQ":
            actual_data_path = self.input_path # e.g. /path/to/BAAI/CCI3-HQ/data
        elif os.path.basename(self.input_path) == "CCI3-HQ":
            actual_data_path = os.path.join(self.input_path, "data") # e.g. /path/to/BAAI/CCI3-HQ
        elif "CCI3-HQ" in self.input_path : # e.g. /path/to/BAAI
             # Attempt to locate data directory more robustly
            potential_path = os.path.join(self.input_path, "CCI3-HQ", "data")
            if os.path.exists(potential_path):
                 actual_data_path = potential_path
            elif os.path.exists(os.path.join(self.input_path, "data")): # If self.input_path is already .../CCI3-HQ/
                 actual_data_path = os.path.join(self.input_path, "data")
            else: # if self.input_path is already .../CCI3-HQ/data
                 actual_data_path = self.input_path

        else: # Fallback if structure is unexpected, assume self.input_path is the data dir
            actual_data_path = self.input_path


        if not os.path.exists(actual_data_path) or not os.path.isdir(actual_data_path):
            print(f"[warning][{self.dataset_name}] 有效数据路径未找到或不是目录: {actual_data_path} (基于原始输入: {self.input_path})")
            return []
        
        # 获取所有jsonl文件的绝对路径
        files = [os.path.join(actual_data_path, f) for f in os.listdir(actual_data_path) if f.endswith(".jsonl") and os.path.isfile(os.path.join(actual_data_path, f))]
        print(f"[log][{self.dataset_name}] 在 {actual_data_path} 中找到 {len(files)} 个jsonl文件")
        return files

    def process_one_line(self, line, fout) -> bool:
        """处理单行数据"""
        try:
            # 解析原始的JSON行
            data = json.loads(line)
            # 获取 "text" 字段的值，如果不存在则默认为空字符串
            text = data.get("text", "")
            
            # # 对文本进行中文处理，例如全角半角统一、unicode统一等
            # processed_text = self.zh_process(text) # 用户之前注释掉了
            
            # # 对处理后的文本进行质量检查，例如字数是否过少
            # if not self.quality_assurance(processed_text): # 用户之前注释掉了
            #     # 如果质量不合格，则跳过这一行，返回False
            #     return False
                
            # 构建输出数据，这里我们只保留 "text" 字段
            output = {"text": text} # 使用原始的 text
            # 将处理后的数据（只包含text字段）以JSON格式写入输出文件
            fout.write(json.dumps(output, ensure_ascii=False) + "\n")
            # 表示成功处理了这一行
            return True
            
        except Exception as e:
            # 如果在处理过程中发生任何错误（例如JSON解析失败）
            # 打印错误信息
            # 注意：此处的print将来自工作进程
            print(f"[exception][{self.dataset_name}] 行处理错误: {str(e)} 内容: '{line[:100]}...'") # 打印部分行内容帮助定位
            # 返回False表示处理失败
            return False

class wiki_EN_FormatHandler(FormatHandler):
    
    """处理wiki_EN数据集（parquet格式）
    示例数据格式：
    {
        "id": 1,
        "url": "https://en.wikipedia.org/wiki/Main_Page",
        "title": "Main Page",
        "text": "Welcome to the Main Page!"
    }
    """

    def __init__(self, input_path, output_path, dataset_name):
        super().__init__(input_path, output_path, dataset_name)
    
    
    def get_file_list(self) -> list:
        """获取输入路径下及其所有子目录中的parquet文件"""
        all_files = []
        
        def traverse_directory(dir_path):
            try:
                for item in os.listdir(dir_path):
                    item_path = os.path.join(dir_path, item)
                    if os.path.isdir(item_path):
                        # 递归搜索子目录
                        traverse_directory(item_path)
                    elif item.endswith(".parquet"):
                        # 添加找到的parquet文件
                        all_files.append(item_path)
            except Exception as e:
                print(f"[warning] 读取目录 {dir_path} 时出错: {e}")
                
        # 从根目录开始递归搜索
        traverse_directory(self.input_path)
        print(f"[log][{self.dataset_name}] 在所有子目录中找到{len(all_files)}个parquet文件")
        return all_files

    def process_one_file(self, file_path):
        line_count = 0
        jump_count = 0
        
        with open(self.output_path, "a") as fout:
            try:
                # 读取parquet文件
                table = pq.read_table(file_path)
                data = table.to_pydict()
                
                # 检查必要的列是否存在
                if "text" not in data:
                    print(f"[warning] 文件 {file_path} 中找不到text列")
                    return 0, 0
                    
                # 获取各列数据
                texts = data["text"]
                ids = data.get("id", [0] * len(texts))
                urls = data.get("url", ["unknown"] * len(texts))
                titles = data.get("title", ["unknown"] * len(texts))
                
                # 处理每一行数据
                total_num = len(texts)
                for i in range(total_num):
                    line_count += 1
                    
                    # 处理文本
                    #text = self.zh_process(texts[i])
                    
                    # 质量检查 - 使用基本质量过滤以及基于quality_score的附加筛选
                    if not self.quality_assurance(texts):
                        jump_count += 1
                        continue
                    
                    # 构建输出结果
                    try:
                        output = {
                            "text": texts,
                            "id": ids[i] if i < len(ids) else 0,
                            "url": urls[i] if i < len(urls) else "unknown",
                            "title": titles[i] if i < len(titles) else "unknown"
                        }
                        
                        fout.write(json.dumps(output, ensure_ascii=False) + "\n")
                    except Exception as e:
                        print(f"[exception][{self.dataset_name}] 处理第 {i} 行时出错: {e}")
                        jump_count += 1
                        
                    # 打印处理进度
                    if line_count % 10000 == 0:
                        print(f"[progress][{self.dataset_name}] 文件 {os.path.basename(file_path)} 已处理 {line_count} 行，成功 {line_count - jump_count} 行")
                        
            except Exception as e:
                print(f"[exception][{self.dataset_name}] 处理文件 {file_path} 失败: {e}")
                print(traceback.format_exc())
                
        return line_count, jump_count


class Dolma_EN_FormatHandler(FormatHandler):
    
    """处理wiki_EN数据集（JSON格式）
     只保留原来的text列

    """

    def __init__(self, input_path, output_path, dataset_name):
        super().__init__(input_path, output_path, dataset_name)
    
    
    def get_file_list(self) -> list:
        """获取输入路径下及其所有子目录中的JSON文件"""
        all_files = []
        
        def traverse_directory(dir_path):
            try:
                for item in os.listdir(dir_path):
                    item_path = os.path.join(dir_path, item)
                    if os.path.isdir(item_path):
                        # 递归搜索子目录
                        traverse_directory(item_path)
                    elif item.endswith(".json"):
                        # 添加找到的JSON文件
                        all_files.append(item_path)
            except Exception as e:
                print(f"[warning] 读取目录 {dir_path} 时出错: {e}")
                
        # 从根目录开始递归搜索
        traverse_directory(self.input_path)
        print(f"[log][{self.dataset_name}] 在所有子目录中找到{len(all_files)}个JSON文件")
        return all_files

    def process_one_file(self, file_path):
        line_count = 0
        jump_count = 0
        
        with open(self.output_path, "a", encoding='utf-8') as fout:
            try:
                # 读取JSON文件
                with open(file_path, "r", encoding='utf-8') as fin:
                    # 支持两种JSON格式：单个对象或对象数组
                    content = fin.read().strip()
                    if content.startswith('['):
                        # JSON数组格式
                        data_list = json.loads(content)
                    else:
                        # 单个JSON对象格式或每行一个JSON对象
                        lines = content.split('\n')
                        data_list = []
                        for line in lines:
                            if line.strip():
                                try:
                                    data_list.append(json.loads(line))
                                except json.JSONDecodeError:
                                    continue
                        
                        # 如果解析失败，尝试作为单个JSON对象解析
                        if not data_list:
                            data_list = [json.loads(content)]
                
                # 处理每一条数据
                for data_item in data_list:
                    line_count += 1
                    
                    # 检查text字段是否存在
                    if "text" not in data_item or not data_item["text"]:
                        jump_count += 1
                        continue
                    
                    text = data_item["text"]
                    
                    # 质量检查
                    if not self.quality_assurance(text):
                        jump_count += 1
                        continue
                    
                    # 构建输出结果（只保留text列）
                    try:
                        output = {
                            "text": text
                        }
                        
                        fout.write(json.dumps(output, ensure_ascii=False) + "\n")
                    except Exception as e:
                        print(f"[exception][{self.dataset_name}] 处理数据时出错: {e}")
                        jump_count += 1
                        
                    # 打印处理进度
                    if line_count % 10000 == 0:
                        print(f"[progress][{self.dataset_name}] 文件 {os.path.basename(file_path)} 已处理 {line_count} 行，成功 {line_count - jump_count} 行")
                        
            except Exception as e:
                print(f"[exception][{self.dataset_name}] 处理文件 {file_path} 失败: {e}")
                print(traceback.format_exc())
                
        return line_count, jump_count


def test_run():
    """简单测试"""
    script_directory = os.path.dirname(os.path.realpath(__file__))
    work_directory = script_directory + "/../download_data/"
    os.chdir(work_directory)
    output_path_root = work_directory + "/output"
    if not os.path.exists(output_path_root):
        os.makedirs(output_path_root)
    
    dataset_handler = {
        "chinese_fine_web_edu": ChineseFineWebEduFormatHandler,
        "cc_i3_HQ": CCI3_HQFormatHandler,
        "baidu_baike": BaiduBaikeFormatHandler,
        "wiki_en": wiki_EN_FormatHandler,
        "dolma_en": Dolma_EN_FormatHandler,
    }

    for dataset_name, Handler in dataset_handler.items():
        input_path = dataset_name
        if dataset_name == "BELLE_conversations":
            input_path = "BELLE"
        output_path = output_path_root + "/processed_{}.jsonl".format(dataset_name)
        if os.path.exists(output_path):
            os.remove(output_path)
        fh = Handler(input_path, output_path, dataset_name)
        fh.process_all()


def main_run():
    input_path_root = "输入数据路径"
    output_path_root = "输出数据路径"
    if not os.path.exists(output_path_root):
        os.makedirs(output_path_root)

    dataset_process_info = {
        #"baidu_baike": (input_path_root + "/fq980207/563w_baidubaike", BaiduBaikeFormatHandler),  #* 16G
        #"wiki_en": (input_path_root + "/swift/wikipedia/data", wiki_EN_FormatHandler),
        #"chinese_fine_web_edu": (input_path_root + "/opencsg/chinese-fineweb-edu/data", ChineseFineWebEduFormatHandler),  #* 122G
        #"cc_i3_HQ": (input_path_root + "/BAAI/CCI3-HQ/data", CCI3_HQFormatHandler),
        "dolma_en": (input_path_root + "/swift/dolma", Dolma_EN_FormatHandler),
    }

    for dataset_name, info in dataset_process_info.items():
        input_path = info[0]
        Handler = info[1]
        output_path = output_path_root + "/processed_{}.jsonl".format(dataset_name)
        if os.path.exists(output_path):
            os.remove(output_path)
        fh = Handler(input_path, output_path, dataset_name)
        fh.process_all(num_processes=40)

if __name__ == "__main__":
    test_mode = False
    if test_mode:
        test_run()
    else: 
        main_run()



    


    

    

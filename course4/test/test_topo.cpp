//
// Created by fss on 23-6-25.
//
#include "runtime/ir.h"
#include "runtime/runtime_ir.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <string>

std::string bin_path("/ssd2/luzhenyu/szluyu99/kuiperdatawhale/course4/model_file/resnet18_batch1.pnnx.bin");
std::string param_path("/ssd2/luzhenyu/szluyu99/kuiperdatawhale/course4/model_file/resnet18_batch1.param");

/*
测试神经网络图的拓扑结构
初始化RuntimeGraph并构建计算图
遍历拓扑队列中的算子，打印每个算子的类型和名称
验证图初始化是否成功
*/
TEST(test_ir, topo) {
  using namespace kuiper_infer;
  RuntimeGraph graph(param_path, bin_path);
  const bool init_success = graph.Init();
  ASSERT_EQ(init_success, true);
  graph.Build("pnnx_input_0", "pnnx_output_0");
  const auto &topo_queues = graph.get_topo_queues();

  int index = 0;
  for (const auto &operator_ : topo_queues) {
    LOG(INFO) << "Index: " << index << " Type: " << operator_->type
              << " Name: " << operator_->name;
    index += 1;
  }
}

/*
测试算子的构建输出
类似topo测试但只打印算子名称
验证图状态从初始化到构建完成的过程
*/
TEST(test_ir, build_output_ops) {
  using namespace kuiper_infer;
  RuntimeGraph graph(param_path, bin_path);
  const bool init_success = graph.Init();
  ASSERT_EQ(init_success, true);
  graph.Build("pnnx_input_0", "pnnx_output_0");
  const auto &topo_queues = graph.get_topo_queues();

  int index = 0;
  for (const auto &operator_ : topo_queues) {
    LOG(INFO) << "Index: " << index << " Name: " << operator_->name;
    index += 1;
  }
}

/*
深入测试算子的输出连接关系
打印每个算子的名称及其输出算子列表
展示算子之间的连接关系
*/
TEST(test_ir, build_output_ops2) {
  using namespace kuiper_infer;
  RuntimeGraph graph(param_path, bin_path);
  const bool init_success = graph.Init();
  ASSERT_EQ(init_success, true);
  graph.Build("pnnx_input_0", "pnnx_output_0");
  const auto &topo_queues = graph.get_topo_queues();

  int index = 0;
  for (const auto &operator_ : topo_queues) {
    LOG(INFO) << "operator name: " << operator_->name;
    for (const auto &pair : operator_->output_operators) {
      LOG(INFO) << "output: " << pair.first;
    }
    LOG(INFO) << "-------------------------";
    index += 1;
  }
}

/*
测试图的构建状态转换
验证图状态从NotInit(-2)到Init(-1)再到Build(0)的转换过程
确保状态机正确工作
*/
TEST(test_ir, build1_status) {
  using namespace kuiper_infer;
  RuntimeGraph graph(param_path, bin_path);
  ASSERT_EQ(int(graph.graph_state()), -2);
  const bool init_success = graph.Init();
  ASSERT_EQ(init_success, true);
  ASSERT_EQ(int(graph.graph_state()), -1);
  graph.Build("pnnx_input_0", "pnnx_output_0");
  ASSERT_EQ(int(graph.graph_state()), 0);
}

/*
测试算子的输出张量信息
打印每个算子的输出张量的维度信息(batch, channel, height, width)
验证神经网络中间结果的维度是否正确
*/
TEST(test_ir, build1_output_tensors) {
  using namespace kuiper_infer;
  RuntimeGraph graph(param_path, bin_path);
  ASSERT_EQ(int(graph.graph_state()), -2);
  const bool init_success = graph.Init();
  ASSERT_EQ(init_success, true);
  ASSERT_EQ(int(graph.graph_state()), -1);
  graph.Build("pnnx_input_0", "pnnx_output_0");
  ASSERT_EQ(int(graph.graph_state()), 0);

  const auto &ops = graph.operators();
  for (const auto &op : ops) {
    LOG(INFO) << op->name;
    // 打印op输出空间的张量
    const auto &operand = op->output_operands;
    if (!operand || operand->datas.empty()) {
      continue;
    }
    const uint32_t batch_size = operand->datas.size();
    LOG(INFO) << "batch: " << batch_size;

    for (uint32_t i = 0; i < batch_size; ++i) {
      const auto &data = operand->datas.at(i);
      LOG(INFO) << "channel: " << data->channels()
                << " height: " << data->rows() << " cols: " << data->cols();
    }
  }
}

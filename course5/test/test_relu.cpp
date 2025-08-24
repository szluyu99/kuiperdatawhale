//
// Created by fss on 23-6-25.
//
#include "layer/abstract/layer_factory.hpp"
#include <gtest/gtest.h>
using namespace kuiper_infer;

static LayerRegisterer::CreateRegistry *RegistryGlobal() {
  static LayerRegisterer::CreateRegistry *kRegistry = new LayerRegisterer::CreateRegistry();
  CHECK(kRegistry != nullptr) << "Global layer register init failed!";
  return kRegistry;
}

/*
测试全局层注册表的单例特性
验证多次获取的 RegistryGlobal() 返回相同指针
确保全局注册表是唯一的
*/
TEST(test_registry, registry1) {
  using namespace kuiper_infer;
  LayerRegisterer::CreateRegistry *registry1 = RegistryGlobal();
  LayerRegisterer::CreateRegistry *registry2 = RegistryGlobal();

  LayerRegisterer::CreateRegistry *registry3 = RegistryGlobal();
  LayerRegisterer::CreateRegistry *registry4 = RegistryGlobal();
  float *a = new float{3};
  float *b = new float{4};
  ASSERT_EQ(registry1, registry2);
}

ParseParameterAttrStatus MyTestCreator(
    const std::shared_ptr<RuntimeOperator> &op,
    std::shared_ptr<Layer> &layer) {

  layer = std::make_shared<Layer>("test_layer");
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

/*
测试 LayerRegisterer 注册功能
验证注册表相等性
测试自定义层类型 "test_type" 的注册和查找
检查注册表大小和内容
*/
TEST(test_registry, registry2) {
  using namespace kuiper_infer;
  LayerRegisterer::CreateRegistry registry1 = LayerRegisterer::Registry();
  LayerRegisterer::CreateRegistry registry2 = LayerRegisterer::Registry();
  ASSERT_EQ(registry1, registry2);
  LayerRegisterer::RegisterCreator("test_type", MyTestCreator);
  LayerRegisterer::CreateRegistry registry3 = LayerRegisterer::Registry();
  ASSERT_EQ(registry3.size(), 3);
  ASSERT_NE(registry3.find("test_type"), registry3.end());
}

/*
测试层创建功能
注册 "test_type_1" 层创建器
验证能成功创建该类型的层实例
检查从 nullptr 到有效层的转变
*/
TEST(test_registry, create_layer) {
  // 注册了一个test_type_1算子
  LayerRegisterer::RegisterCreator("test_type_1", MyTestCreator);
  std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
  op->type = "test_type_1";
  std::shared_ptr<Layer> layer;
  ASSERT_EQ(layer, nullptr);
  layer = LayerRegisterer::CreateLayer(op);
  ASSERT_NE(layer, nullptr);
}

/*
测试 LayerRegistererWrapper 辅助类
使用包装器注册 "test_type_2"
验证能成功创建该类型的层
*/
TEST(test_registry, create_layer_util) {
  LayerRegistererWrapper kReluGetInstance("test_type_2", MyTestCreator);
  std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
  op->type = "test_type_2";
  std::shared_ptr<Layer> layer;
  ASSERT_EQ(layer, nullptr);
  layer = LayerRegisterer::CreateLayer(op);
  ASSERT_NE(layer, nullptr);
}

/*
测试 ReLU 层的前向传播
创建 "nn.ReLU" 类型的层
生成随机输入张量(3通道,4x4)
执行前向传播
输出 ReLU 激活后的结果
验证输入输出转换
*/
TEST(test_registry, create_layer_reluforward) {
  std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
  op->type = "nn.ReLU";
  std::shared_ptr<Layer> layer;
  ASSERT_EQ(layer, nullptr);
  layer = LayerRegisterer::CreateLayer(op);
  ASSERT_NE(layer, nullptr);

  sftensor input_tensor = std::make_shared<ftensor>(3, 4, 4);
  input_tensor->Rand();
  input_tensor->data() -= 0.5f;

  LOG(INFO) << input_tensor->data();

  std::vector<sftensor> inputs(1);
  std::vector<sftensor> outputs(1);
  inputs.at(0) = input_tensor;
  layer->Forward(inputs, outputs);

  for (const auto &output : outputs) {
    output->Show();
  }
}
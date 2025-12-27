import unittest
from unittest.mock import Mock, patch, MagicMock, call
import torch
import numpy as np
from hallucination_reduction.generator import (
    load_generator,
    build_rag_prompt,
    generate_answer,
    sft_finetune_generator
)


class TestLoadGenerator(unittest.TestCase):
    
    @patch('hallucination_reduction.generator.AutoModelForCausalLM')
    @patch('hallucination_reduction.generator.AutoTokenizer')
    def test_load_generator_with_pad_token(self, mock_tokenizer_class, mock_model_class):
        """Test loading generator when tokenizer has pad_token"""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "[PAD]"
        mock_tokenizer.eos_token = "[EOS]"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        tokenizer, model = load_generator("test-model", "cuda")
        
        mock_tokenizer_class.from_pretrained.assert_called_once_with("test-model")
        mock_model_class.from_pretrained.assert_called_once_with(
            "test-model",
            device_map="auto",
            dtype=torch.bfloat16,
            load_in_4bit=True
        )
        self.assertEqual(tokenizer.pad_token, "[PAD]")
        model.eval.assert_called_once()
    
    @patch('hallucination_reduction.generator.AutoModelForCausalLM')
    @patch('hallucination_reduction.generator.AutoTokenizer')
    def test_load_generator_without_pad_token(self, mock_tokenizer_class, mock_model_class):
        """Test loading generator when tokenizer lacks pad_token"""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "[EOS]"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        tokenizer, model = load_generator("test-model", "cpu")
        
        self.assertEqual(tokenizer.pad_token, "[EOS]")
        model.eval.assert_called_once()


class TestBuildRagPrompt(unittest.TestCase):
    
    def test_build_rag_prompt_single_doc(self):
        """Test building prompt with single document"""
        question = "What is the capital of France?"
        docs = ["Paris is the capital of France."]
        
        prompt = build_rag_prompt(question, docs)
        
        self.assertIn("### Context:", prompt)
        self.assertIn("[1] Paris is the capital of France.", prompt)
        self.assertIn("### Question:", prompt)
        self.assertIn(question, prompt)
        self.assertIn("### Answer:", prompt)
    
    def test_build_rag_prompt_multiple_docs(self):
        """Test building prompt with multiple documents"""
        question = "What are some famous landmarks?"
        docs = [
            "The Eiffel Tower is in Paris.",
            "The Statue of Liberty is in New York.",
            "The Colosseum is in Rome."
        ]
        
        prompt = build_rag_prompt(question, docs)
        
        self.assertIn("[1] The Eiffel Tower is in Paris.", prompt)
        self.assertIn("[2] The Statue of Liberty is in New York.", prompt)
        self.assertIn("[3] The Colosseum is in Rome.", prompt)
        self.assertIn(question, prompt)
    
    def test_build_rag_prompt_empty_docs(self):
        """Test building prompt with no documents"""
        question = "What is AI?"
        docs = []
        
        prompt = build_rag_prompt(question, docs)
        
        self.assertIn("### Context:", prompt)
        self.assertIn("### Question:", prompt)
        self.assertIn(question, prompt)
        self.assertIn("### Answer:", prompt)


class TestGenerateAnswer(unittest.TestCase):
    
    def test_generate_answer_single_sequence(self):
        """Test generating single answer"""
        mock_generator = Mock()
        mock_tokenizer = Mock()
        
        # Mock tokenizer input - needs to return an object with .to() method
        mock_inputs = MagicMock()
        mock_inputs.__getitem__.side_effect = lambda key: {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }[key]
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer.eos_token_id = 0
        
        # Mock generator output
        mock_generator.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
        mock_tokenizer.decode.return_value = "This is the generated answer."
        
        prompt = "Test prompt"
        texts = generate_answer(mock_generator, mock_tokenizer, prompt, num_return_sequences=1)
        
        self.assertEqual(len(texts), 1)
        self.assertEqual(texts[0], "This is the generated answer.")
        mock_generator.generate.assert_called_once()
    
    def test_generate_answer_multiple_sequences(self):
        """Test generating multiple answers"""
        mock_generator = Mock()
        mock_tokenizer = Mock()
        
        mock_inputs = MagicMock()
        mock_inputs.__getitem__.side_effect = lambda key: {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }[key]
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer.eos_token_id = 0
        
        mock_generator.generate.return_value = torch.tensor([
            [1, 2, 3, 4, 5],
            [1, 2, 3, 6, 7]
        ])
        mock_tokenizer.decode.side_effect = ["Answer 1", "Answer 2"]
        
        prompt = "Test prompt"
        texts = generate_answer(mock_generator, mock_tokenizer, prompt, num_return_sequences=2)
        
        self.assertEqual(len(texts), 2)
        self.assertEqual(texts[0], "Answer 1")
        self.assertEqual(texts[1], "Answer 2")
    
    def test_generate_answer_parameters(self):
        """Test that generation uses correct parameters"""
        mock_generator = Mock()
        mock_tokenizer = Mock()
        
        mock_inputs = MagicMock()
        mock_inputs.__getitem__.side_effect = lambda key: {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }[key]
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer.eos_token_id = 0
        mock_generator.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        mock_tokenizer.decode.return_value = "Answer"
        
        generate_answer(
            mock_generator, 
            mock_tokenizer, 
            "prompt",
            max_new_tokens=100,
            min_new_tokens=10,
            temperature=0.5
        )
        
        call_kwargs = mock_generator.generate.call_args[1]
        self.assertEqual(call_kwargs['max_new_tokens'], 100)
        self.assertEqual(call_kwargs['min_new_tokens'], 10)
        self.assertEqual(call_kwargs['temperature'], 0.5)
        self.assertTrue(call_kwargs['do_sample'])
        self.assertEqual(call_kwargs['top_k'], 50)


class TestSFTFinetuneGenerator(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_generator = Mock()
        self.mock_tokenizer = Mock()
        
        # Create mock QA pairs
        self.qa_pairs = []
        for i in range(3):
            qa = Mock()
            qa.question = f"Question {i}?"
            qa.supporting_passages = [f"Document {i}"]
            qa.answer = f"Answer {i}"
            self.qa_pairs.append(qa)
    
    @patch('hallucination_reduction.generator.DataLoader')
    @patch('hallucination_reduction.generator.optim.AdamW')
    def test_sft_finetune_basic(self, mock_adamw, mock_dataloader_class):
        """Test basic fine-tuning functionality"""
        # Mock tokenizer
        self.mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        }
        
        # Mock dataloader - use MagicMock for __iter__
        mock_dataloader = MagicMock()
        mock_batch = (torch.tensor([[1, 2, 3]]), torch.tensor([[1, 1, 1]]))
        mock_dataloader.__iter__.return_value = iter([mock_batch])
        mock_dataloader_class.return_value = mock_dataloader
        
        # Mock optimizer
        mock_optimizer = Mock()
        mock_adamw.return_value = mock_optimizer
        
        # Mock model output - loss needs requires_grad=True for backward
        mock_output = Mock()
        mock_output.loss = torch.tensor(0.5, requires_grad=True)
        self.mock_generator.return_value = mock_output
        self.mock_generator.parameters.return_value = []
        
        result = sft_finetune_generator(
            self.mock_generator,
            self.mock_tokenizer,
            self.qa_pairs,
            epochs=2,
            batch_size=1,
            lr=0.001
        )
        
        self.mock_generator.train.assert_called_once()
        self.mock_generator.eval.assert_called_once()
        self.assertEqual(result, self.mock_generator)
    
    @patch('hallucination_reduction.generator.DataLoader')
    @patch('hallucination_reduction.generator.optim.AdamW')
    def test_sft_finetune_optimizer_steps(self, mock_adamw, mock_dataloader_class):
        """Test that optimizer steps are called correctly"""
        self.mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        mock_dataloader = MagicMock()
        mock_batch = (torch.tensor([[1, 2, 3]]), torch.tensor([[1, 1, 1]]))
        mock_dataloader.__iter__.return_value = iter([mock_batch, mock_batch])
        mock_dataloader_class.return_value = mock_dataloader
        
        mock_optimizer = Mock()
        mock_adamw.return_value = mock_optimizer
        
        mock_output = Mock()
        mock_output.loss = torch.tensor(0.5, requires_grad=True)
        self.mock_generator.return_value = mock_output
        self.mock_generator.parameters.return_value = []
        
        sft_finetune_generator(
            self.mock_generator,
            self.mock_tokenizer,
            [self.qa_pairs[0]],
            epochs=1
        )
        
        # Should be called twice (once per batch)
        self.assertEqual(mock_optimizer.zero_grad.call_count, 2)
        self.assertEqual(mock_optimizer.step.call_count, 2)
    
    @patch('hallucination_reduction.generator.DataLoader')
    @patch('hallucination_reduction.generator.optim.AdamW')
    @patch('builtins.print')
    def test_sft_finetune_loss_logging(self, mock_print, mock_adamw, mock_dataloader_class):
        """Test that losses are logged correctly"""
        self.mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        mock_dataloader = MagicMock()
        mock_batch = (torch.tensor([[1, 2, 3]]), torch.tensor([[1, 1, 1]]))
        mock_dataloader.__iter__.return_value = iter([mock_batch])
        mock_dataloader_class.return_value = mock_dataloader
        
        mock_optimizer = Mock()
        mock_adamw.return_value = mock_optimizer
        
        mock_output = Mock()
        mock_output.loss = torch.tensor(0.5, requires_grad=True)
        self.mock_generator.return_value = mock_output
        self.mock_generator.parameters.return_value = []
        
        sft_finetune_generator(
            self.mock_generator,
            self.mock_tokenizer,
            [self.qa_pairs[0]],
            epochs=2
        )
        
        # Should print loss for each epoch
        self.assertEqual(mock_print.call_count, 2)
        first_call = mock_print.call_args_list[0][0][0]
        self.assertIn("SFT epoch 1/2", first_call)
        self.assertIn("loss=0.5000", first_call)


class TestIntegration(unittest.TestCase):
    """Integration tests for the generator module"""
    
    def test_prompt_generation_and_answer_flow(self):
        """Test that prompts can be built and used for generation"""
        question = "What is machine learning?"
        docs = ["Machine learning is a subset of AI.", "It involves training models on data."]
        
        prompt = build_rag_prompt(question, docs)
        
        self.assertIn(question, prompt)
        for doc in docs:
            self.assertIn(doc, prompt)
        
        # Verify prompt structure
        parts = prompt.split("###")
        self.assertEqual(len(parts), 4)  # Empty, Context, Question, Answer


if __name__ == '__main__':
    unittest.main()
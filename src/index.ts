import * as tf from '@tensorflow/tfjs-node';
// import { pipeline } from '@huggingface/transformers';

// Load the converted model
const model = await tf.loadGraphModel('file://distilbert_ner_tfjs/model.json')

console.log(model.metadata)

// // Use a tokenizer that works with the converted model
// const tokenizer = ... // You might need to implement or find a JS tokenizer compatible with the Hugging Face tokenizer

// // Run inference (you'll need to prepare the tokenization step similar to Hugging Face's pipeline)
// const example = "My name is Wolfgang and I live in Berlin";

// // Process the input and run the model prediction
// const tokens = tokenizer(example);  // Tokenize using your JavaScript tokenizer
// const output = await model.executeAsync(tokens);  // Run the model with tokenized input

// console.log(output);

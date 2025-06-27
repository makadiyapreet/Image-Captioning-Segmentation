# src/captioning/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Optional
import math

class AdaptiveAttention(nn.Module):
    """Adaptive attention mechanism for image captioning"""
    
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super().__init__()
        
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # Adaptive gate
        self.sentinel_gate = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, encoder_out, decoder_hidden):
        """
        encoder_out: (batch_size, num_pixels, encoder_dim)
        decoder_hidden: (batch_size, decoder_dim)
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        
        alpha = F.softmax(att, dim=1)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        
        # Adaptive gate
        gate = self.sigmoid(self.sentinel_gate(decoder_hidden))  # (batch_size, encoder_dim)
        attention_weighted_encoding = gate * attention_weighted_encoding
        
        return attention_weighted_encoding, alpha

class ImageEncoder(nn.Module):
    """ResNet-based image encoder with spatial features"""
    
    def __init__(self, encoded_image_size: int = 14, fine_tune: bool = True):
        super().__init__()
        
        # Load pretrained ResNet-101
        resnet = models.resnet101(pretrained=True)
        
        # Remove last two layers (avgpool and fc)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        # Adaptive pooling to get fixed size feature maps
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        
        # Feature dimension (ResNet-101 has 2048 features)
        self.encoder_dim = 2048
        
        # Fine-tuning
        if fine_tune:
            # Fine-tune only the last few layers
            for param in self.resnet.parameters():
                param.requires_grad = False
            
            # Enable gradients for the last few layers
            for module in list(self.resnet.children())[-3:]:
                for param in module.parameters():
                    param.requires_grad = True
    
    def forward(self, images):
        """
        images: (batch_size, 3, image_size, image_size)
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

class CaptionDecoder(nn.Module):
    """LSTM-based decoder with attention"""
    
    def __init__(self, 
                 attention_dim: int,
                 embed_dim: int,
                 decoder_dim: int,
                 vocab_size: int,
                 encoder_dim: int = 2048,
                 dropout: float = 0.5):
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        
        # Attention mechanism
        self.attention = AdaptiveAttention(encoder_dim, decoder_dim, attention_dim)
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
        # LSTM cell
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        
        # Linear layer to find initial hidden state
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        
        # Linear layer to find scores over vocabulary
        self.fc = nn.Linear(decoder_dim, vocab_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights"""
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
    
    def load_pretrained_embeddings(self, embeddings):
        """Load pre-trained word embeddings"""
        self.embedding.weight = nn.Parameter(embeddings)
    
    def fine_tune_embeddings(self, fine_tune=True):
        """Allow fine-tuning of embedding layer"""
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune
    
    def init_hidden_state(self, encoder_out):
        """Initialize hidden and cell states"""
        mean_encoder_out = encoder_out.mean(dim=1)  # (batch_size, encoder_dim)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)  # (batch_size, decoder_dim)
        return h, c
    
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation
        
        encoder_out: (batch_size, enc_image_size, enc_image_size, encoder_dim)
        encoded_captions: (batch_size, max_caption_length)
        caption_lengths: (batch_size, 1)
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        
        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)
        
        # Sort input data by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        
        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        
        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        decode_lengths = (caption_lengths - 1).tolist()
        
        # Create tensors to hold word prediction scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)
        
        # At each time-step, decode by attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                              h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout_layer(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
        
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

class ImageCaptioningModel(nn.Module):
    """Complete Image Captioning Model"""
    
    def __init__(self, 
                 attention_dim: int = 512,
                 embed_dim: int = 512,
                 decoder_dim: int = 512,
                 vocab_size: int = 10000,
                 encoder_dim: int = 2048,
                 dropout: float = 0.5,
                 device: str = 'cpu'):
        super().__init__()
        
        self.device = device
        self.vocab_size = vocab_size
        
        # Encoder and Decoder
        self.encoder = ImageEncoder()
        self.decoder = CaptionDecoder(attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim, dropout)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, images, captions, caption_lengths):
        """Forward pass"""
        # Encode images
        encoder_out = self.encoder(images)
        
        # Decode captions
        predictions, targets, decode_lengths, alphas, sort_ind = self.decoder(
            encoder_out, captions, caption_lengths)
        
        return predictions, targets, decode_lengths, alphas, sort_ind
    
    def generate_caption(self, image, vocabulary, max_length=20, beam_size=3):
        """Generate caption for a single image using beam search"""
        self.eval()
        
        with torch.no_grad():
            # Encode image
            encoder_out = self.encoder(image.unsqueeze(0))  # (1, enc_image_size, enc_image_size, encoder_dim)
            encoder_dim = encoder_out.size(-1)
            
            # Flatten encoding
            encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
            num_pixels = encoder_out.size(1)
            
            # We'll treat the problem as having a batch size of k
            encoder_out = encoder_out.expand(beam_size, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
            
            # Tensor to store top k previous words at each step; now they're just <start>
            k_prev_words = torch.LongTensor([[vocabulary.word2idx['<START>']]] * beam_size).to(self.device)  # (k, 1)
            
            # Tensor to store top k sequences; now they're just <start>
            seqs = k_prev_words  # (k, 1)
            
            # Tensor to store top k sequences' scores; now they're just 0
            top_k_scores = torch.zeros(beam_size, 1).to(self.device)  # (k, 1)
            
            # Lists to store completed sequences and scores
            complete_seqs = list()
            complete_seqs_scores = list()
            
            # Start decoding
            step = 1
            h, c = self.decoder.init_hidden_state(encoder_out)
            
            # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
            while True:
                embeddings = self.decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
                
                awe, _ = self.decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
                
                h, c = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
                
                scores = self.decoder.fc(h)  # (s, vocab_size)
                scores = F.log_softmax(scores, dim=1)
                
                # Add
                scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
                
                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(beam_size, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(beam_size, 0, True, True)  # (s)
                
                # Convert unrolled indices to actual indices of scores
                prev_word_inds = top_k_words // self.vocab_size  # (s)
                next_word_inds = top_k_words % self.vocab_size  # (s)
                
                # Add new words to sequences
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
                
                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                 next_word != vocabulary.word2idx['<END>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
                
                # Set aside complete sequences
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                beam_size -= len(complete_inds)  # reduce beam length accordingly
                
                # Proceed with incomplete sequences
                if beam_size == 0:
                    break
                seqs = seqs[incomplete_inds]
                h = h[prev_word_inds[incomplete_inds]]
                c = c[prev_word_inds[incomplete_inds]]
                encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
                
                # Break if things have been going on too long
                if step > max_length:
                    break
                step += 1
            
            # Choose the sequence with the best score
            if len(complete_seqs_scores) > 0:
                i = complete_seqs_scores.index(max(complete_seqs_scores))
                seq = complete_seqs[i]
            else:
                seq = seqs[0].tolist()
            
            # Convert to words
            caption = vocabulary.decode_caption(seq)
            
        return caption

if __name__ == "__main__":
    # Test the model
    print("🧪 Testing Image Captioning Model...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Create model
    model = ImageCaptioningModel(
        attention_dim=512,
        embed_dim=512,
        decoder_dim=512,
        vocab_size=1000,
        device=device
    ).to(device)
    
    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    captions = torch.randint(0, 1000, (batch_size, 15)).to(device)
    caption_lengths = torch.tensor([15, 12]).unsqueeze(1).to(device)
    
    try:
        predictions, targets, decode_lengths, alphas, sort_ind = model(images, captions, caption_lengths)
        print(f"✅ Model test passed!")
        print(f"📊 Predictions shape: {predictions.shape}")
        print(f"📊 Alphas shape: {alphas.shape}")
    except Exception as e:
        print(f"❌ Model test failed: {e}")
    
    print("🎯 Model ready for training!")
import re
import json
import logging
import time
import os
import math
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple, Optional

from swift.plugin.orm import ORM, orms

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepfakeComplexReward(ORM):
    def __init__(self):
        super().__init__()
        # Default log path, use relative path to ensure logs are written to the correct location
        default_log_path = "./grpo_reward_log.jsonl"
        self.log_path = os.environ.get("GRPO_REWARD_LOG_PATH", default_log_path)
        
        # Debug log switch (default off)
        self.debug_enabled = os.environ.get("GRPO_DEBUG_LOG", "false").lower() == "true"
        self.debug_log_path = "./debug.log"
    
    def _debug_log(self, location: str, message: str, data: dict):
        """Conditionally write debug log"""
        if not self.debug_enabled:
            return
        try:
            import json as _json
            open(self.debug_log_path, "a").write(_json.dumps({
                "location": location,
                "message": message,
                "data": data,
                "timestamp": time.time()
            }) + "\n")
        except:
            pass
        
    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        """
        Calculates a comprehensive reward for Deepfake detection and localization.
        
        Reward Structure (Total = R_fmt + R_cls + R_bbox_penalty + R_coverage):
        1. Format Reward (max 1.0): Presence of <think>, <description>, <evidence>, <conclusion> tags.
        2. Classification Reward (±2.0): Correct classification from the last sentence.
        3. Bbox Penalty (-1.0): Strict penalty for bbox mismatch (Real predicted but bbox present, Fake predicted but bbox missing).
        4. Coverage Reward (max 0.25): Smooth reward for bbox coverage (intersection area / ground truth area, only for correct classification and fake samples).
        """
        self._debug_log("__call__", "Entry", {
            "completions_len": len(completions),
            "solution_len": len(solution)
        })
        rewards = []
        log_entries = [] # Buffer for log entries
        timestamp = time.time()
        
        # First pass: parse all data
        parsed_data = []
        for idx, (content, sol_str) in enumerate(zip(completions, solution)):
            try:    
                gt_data = self._parse_ground_truth(sol_str)
                if not gt_data:
                    parsed_data.append(None)
                    continue
                
                pred_data = self._parse_prediction(content)
                parsed_data.append({
                    'gt_data': gt_data,
                    'pred_data': pred_data,
                    'content': content,
                    'sol_str': sol_str
                })
            except Exception as e:
                logger.error(f"Error parsing sample {idx}: {e}")
                parsed_data.append(None)

        # Second pass: calculate final reward
        for idx, data in enumerate(parsed_data):
            log_entry = {
                'timestamp': timestamp,
                'completion': completions[idx] if idx < len(completions) else '',
                'solution': solution[idx] if idx < len(solution) else '',
                'reward': 0.0,
                'components': {},
                'error': None
            }
            
            if data is None:
                rewards.append(0.0)
                log_entry['error'] = 'Invalid GT or parsing error'
                log_entries.append(log_entry)
                continue
            
            try:
                gt_data = data['gt_data']
                pred_data = data['pred_data']
                
                # --- 1. Format Reward (max 1.0) ---
                r_format = 0.0
                if gt_data['is_fake']:
                    # Fake samples: 4 basic tags (0.2*4=0.8) + Bbox format (0.2)
                    if pred_data['has_think']: r_format += 0.2
                    if pred_data['has_description']: r_format += 0.2
                    if pred_data['has_evidence']: r_format += 0.2
                    if pred_data['has_conclusion']: r_format += 0.2
                    if pred_data['has_box_tag'] and pred_data['has_object_ref']: r_format += 0.2
                else:
                    # Real samples: 4 basic tags (0.25*4=1.0)
                    if pred_data['has_think']: r_format += 0.25
                    if pred_data['has_description']: r_format += 0.25
                    if pred_data['has_evidence']: r_format += 0.25
                    if pred_data['has_conclusion']: r_format += 0.25
                
                # --- 2. Classification Reward (±2.0) ---
                # Extraction must be successful and result consistent to be correct; extraction failure (None) is error-handled
                is_correct = (pred_data['is_fake'] is not None) and (pred_data['is_fake'] == gt_data['is_fake'])
                r_class = 4.0 if is_correct else -4.0
                
                # --- 3. Bbox Penalty (-1.0) ---
                # Strict penalty: Real predicted but bbox present, or Fake predicted but bbox missing
                r_bbox_penalty = 0.0
                pred_has_bbox = pred_data['bbox'] is not None
                gt_is_fake = gt_data['is_fake']
                
                if gt_is_fake and not pred_has_bbox:
                    # Fake sample but predicted no bbox
                    r_bbox_penalty = -1.0
                elif not gt_is_fake and pred_has_bbox:
                    # Real sample but predicted has bbox
                    r_bbox_penalty = -1.0
                
                # Classification error logic: stop further reward calculation
                if not is_correct:
                    total_reward = r_format + r_class + r_bbox_penalty
                    rewards.append(total_reward)
                    log_entry['reward'] = total_reward
                    log_entry['components'] = {
                        'format': r_format,
                        'class': r_class,
                        'bbox_penalty': r_bbox_penalty
                    }
                    log_entries.append(log_entry)
                    continue

                # --- 4. Coverage Reward (max 0.25) ---
                # Only calculate coverage reward for fake samples with predicted bbox
                # Coverage = intersection area / ground truth area
                r_coverage = 0.0
                if gt_is_fake and pred_has_bbox and gt_data['bbox']:
                    coverage = self.calculate_coverage(pred_data['bbox'], gt_data['bbox'])
                    # Use smooth sigmoid function for stable reward with less noise
                    # When coverage is close to 1, reward is close to 0.25; when close to 0, reward is close to 0
                    # Using sigmoid: 0.25 * (1 / (1 + exp(-k * (coverage - threshold))))
                    # k=10, threshold=0.5 makes reward ~0.125 at coverage=0.5, ~0.2 at coverage=0.7, ~0.05 at coverage=0.1
                    r_coverage = 0.25 * (1.0 / (1.0 + math.exp(-10.0 * (coverage - 0.5))))
                
                # --- Total Reward ---
                total_reward = r_format + r_class + r_bbox_penalty + r_coverage
                rewards.append(total_reward)
                
                # Log entry update
                log_entry['reward'] = total_reward
                log_entry['components'] = {
                    'format': r_format,
                    'class': r_class,
                    'bbox_penalty': r_bbox_penalty,
                    'coverage': r_coverage
                }
                log_entries.append(log_entry)

            except Exception as e:
                rewards.append(0.0)
                log_entry['error'] = str(e)
                log_entries.append(log_entry)
        
        # Batch Write Log
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                for entry in log_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except: pass
            
        return rewards

    def calculate_coverage(self, pred_box, gt_box):
        """Calculate coverage: intersection area of predicted and ground truth boxes divided by ground truth area"""
        # box: [x1, y1, x2, y2]
        # pred_box: predicted box
        # gt_box: ground truth box
        x1_pred, y1_pred, x2_pred, y2_pred = pred_box
        x1_gt, y1_gt, x2_gt, y2_gt = gt_box
        
        # Ensure coordinates are well-formed
        x1_pred, x2_pred = min(x1_pred, x2_pred), max(x1_pred, x2_pred)
        y1_pred, y2_pred = min(y1_pred, y2_pred), max(y1_pred, y2_pred)
        x1_gt, x2_gt = min(x1_gt, x2_gt), max(x1_gt, x2_gt)
        y1_gt, y2_gt = min(y1_gt, y2_gt), max(y1_gt, y2_gt)
        
        # Calculate Intersection
        xi1 = max(x1_pred, x1_gt)
        yi1 = max(y1_pred, y1_gt)
        xi2 = min(x2_pred, x2_gt)
        yi2 = min(y2_pred, y2_gt)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate Ground Truth area
        gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
        
        # Handle edge case for zero gt area
        if gt_area == 0:
            return 0.0
            
        coverage = inter_area / gt_area
        return coverage

    def _parse_ground_truth(self, sol_str: str) -> Optional[Dict]:
        """Parses the ground truth string/object according to the new format."""
        try:
            # 1. Classification check from the last sentence
            lines = sol_str.strip().split('\n')
            last_line = lines[-1].strip() if lines else ""
            if "is a deepfake" in last_line.lower():
                is_fake = True
            elif "is a real image" in last_line.lower():
                is_fake = False
            else:
                return None  # Malformed GT, skip
            
            # 2. Extract components
            def extract_tag(tag, text):
                match = re.search(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
                return match.group(1).strip() if match else ""

            description = extract_tag('description', sol_str)
            evidence = extract_tag('evidence', sol_str)
            conclusion = extract_tag('conclusion', sol_str)
            
            # 3. Extract bbox if fake
            bbox = None
            if is_fake:
                # Match new bbox format: <|box_start|>x1="349" y1="125" x2="635" y2="455"<|box_end|>
                box_match = re.search(r'<\|box_start\|>(.*?)<\|box_end\|>', sol_str)
                if box_match:
                    attrs = box_match.group(1)
                    def get_attr(name, s):
                        m = re.search(rf'{name}=["\']?(\d+)["\']?', s)
                        return int(m.group(1)) if m else None
                    x1 = get_attr('x1', attrs)
                    y1 = get_attr('y1', attrs)
                    x2 = get_attr('x2', attrs)
                    y2 = get_attr('y2', attrs)
                    if None not in [x1, y1, x2, y2]:
                        bbox = [x1, y1, x2, y2]
            
            return {
                'is_fake': is_fake,
                'bbox': bbox,
                'description': description,
                'evidence': evidence,
                'conclusion': conclusion
            }
        except Exception as e:
            logger.error(f"Error parsing ground truth: {e}")
            return None

    def _parse_prediction(self, text: str) -> Dict:
        """Robustly parses the model prediction with new tags and bbox format."""
        result = {
            'has_think': "<think>" in text and "</think>" in text,
            'has_description': "<description>" in text and "</description>" in text,
            'has_evidence': "<evidence>" in text and "</evidence>" in text,
            'has_conclusion': "<conclusion>" in text and "</conclusion>" in text,
            'description_text': "",
            'evidence_text': "",
            'conclusion_text': "",
            'is_fake': None,
            'has_box_tag': "<|box_start|>" in text and "<|box_end|>" in text,
            'has_object_ref': '<|object_ref_start|>"deepfake"<|object_ref_end|>' in text,
            'bbox': None
        }
        
        # 1. Extract tag contents
        def extract_tag(tag, s):
            match = re.search(rf'<{tag}>(.*?)</{tag}>', s, re.DOTALL)
            return match.group(1).strip() if match else ""

        result['description_text'] = extract_tag('description', text)
        result['evidence_text'] = extract_tag('evidence', text)
        result['conclusion_text'] = extract_tag('conclusion', text)
        
        # 2. Classification check from the last line
        lines = text.strip().split('\n')
        result['is_fake'] = None  # Initialize to None, keep if extraction fails
        if lines:
            # Get last non-empty line
            last_lines = [l.strip().lower() for l in lines if l.strip()]
            if last_lines:
                last_line = last_lines[-1]
                if "is a real image" in last_line:
                    result['is_fake'] = False
                elif "is a deepfake" in last_line:
                    result['is_fake'] = True
                else:
                    # Explicitly set to None if last line doesn't contain keywords
                    result['is_fake'] = None
            
        # 3. Bbox extraction if present
        if result['has_box_tag']:
            box_match = re.search(r'<\|box_start\|>(.*?)<\|box_end\|>', text)
            if box_match:
                attrs = box_match.group(1)
                def get_attr(name, s):
                    m = re.search(rf'{name}=["\']?(\d+)["\']?', s)
                    return int(m.group(1)) if m else None
                x1 = get_attr('x1', attrs)
                y1 = get_attr('y1', attrs)
                x2 = get_attr('x2', attrs)
                y2 = get_attr('y2', attrs)
                if None not in [x1, y1, x2, y2]:
                    result['bbox'] = [x1, y1, x2, y2]
                    
        return result


# Register the reward function
orms['deepfake_complex_reward'] = DeepfakeComplexReward

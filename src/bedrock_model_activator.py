import logging
from typing import Any, Dict, List

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class BedrockModelActivator:
    def __init__(self, region_name: str = 'us-east-1'):
        """
        Initialize Bedrock Model Activator

        Args:
            region_name: AWS region to use for Bedrock operations
        """
        self.region_name = region_name
        self.bedrock_client = boto3.client('bedrock', region_name=region_name)

    def list_foundation_models(self) -> List[Dict[str, Any]]:
        """
        List all available foundation models

        Returns:
            List of foundation model summaries
        """
        try:
            logger.info("Retrieving list of foundation models...")
            response = self.bedrock_client.list_foundation_models()
            models = response.get('modelSummaries', [])
            logger.info(f"Found {len(models)} foundation models")
            return models
        except ClientError as e:
            logger.error(f"Error listing foundation models: {e}")
            raise

    def filter_on_demand_models(self, models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter models to only include those supporting ON_DEMAND or INFERENCE_PROFILE

        Args:
            models: List of model summaries from list_foundation_models

        Returns:
            List of filtered models supporting ON_DEMAND or INFERENCE_PROFILE
        """
        filtered_models = []

        for model in models:
            model_id = model.get('modelId')
            inference_types = model.get('inferenceTypesSupported', [])

            # Check if model supports ON_DEMAND or INFERENCE_PROFILE
            if any(inference_type in ['ON_DEMAND', 'INFERENCE_PROFILE']
                   for inference_type in inference_types):
                filtered_models.append(model)
                logger.info(f"Model {model_id} supports: {inference_types}")
            else:
                logger.info(f"Skipping model {model_id} - unsupported inference types: {inference_types}")

        logger.info(f"Filtered to {len(filtered_models)} ON_DEMAND/INFERENCE_PROFILE models")
        return filtered_models

    def check_model_access_status(self, models: List[Dict[str, Any]]) -> tuple[
        List[Dict], List[Dict]
    ]:
        """
        Check which models need access requests using get_foundation_model_availability API

        Args:
            models: List of model summaries from list_foundation_models

        Returns:
            Tuple of (accessible_models, models_needing_access)
        """
        accessible_models = []
        models_needing_access = []

        for model in models:
            model_id = model.get('modelId')

            try:
                logger.info(f"Checking availability for model: {model_id}")
                response = self.bedrock_client.get_foundation_model_availability(
                    modelId=model_id
                )

                agreement_availability = response.get('agreementAvailability', {})
                agreement_status = agreement_availability.get('status', 'UNKNOWN')
                authorization_status = response.get('authorizationStatus', 'UNKNOWN')

                logger.info(f"Model {model_id}: agreement={agreement_status}, authorization={authorization_status}")

                # Models that need activation have agreementAvailability.status == 'NOT_AVAILABLE'
                if agreement_status == 'NOT_AVAILABLE':
                    models_needing_access.append(model)
                else:
                    accessible_models.append(model)

            except ClientError as e:
                logger.error(f"Error checking availability for {model_id}: {e}")
                # If we can't check, assume it needs access
                models_needing_access.append(model)

        logger.info(f"Accessible models: {len(accessible_models)}")
        logger.info(f"Models needing access: {len(models_needing_access)}")

        return accessible_models, models_needing_access

    def get_agreement_offers(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Get agreement offers for a specific model

        Args:
            model_id: The model ID to get offers for

        Returns:
            List of offer details
        """
        try:
            logger.info(f"Getting agreement offers for model: {model_id}")
            response = self.bedrock_client.list_foundation_model_agreement_offers(
                modelId=model_id
            )
            offers = response.get('offers', [])
            logger.info(f"Found {len(offers)} offers for model {model_id}")
            return offers
        except ClientError as e:
            logger.error(f"Error getting agreement offers for {model_id}: {e}")
            return []

    def create_model_agreement(self, model_id: str, offer_token: str) -> bool:
        """
        Create a foundation model agreement

        Args:
            model_id: The model ID to create agreement for
            offer_token: The offer token obtained from list_foundation_model_agreement_offers

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Creating model agreement for {model_id}")
            self.bedrock_client.create_foundation_model_agreement(
                modelId=model_id,
                offerToken=offer_token
            )
            logger.info(f"Successfully created agreement for model {model_id}")
            return True
        except ClientError as e:
            logger.error(f"Error creating agreement for {model_id}: {e}")
            return False

    def activate_all_models(self) -> Dict[str, Any]:
        """
        Main method to activate all available Bedrock models

        Returns:
            Summary of activation results
        """
        logger.info(f"Starting model activation process in region: {self.region_name}")

        # Step 1: List all foundation models
        all_models = self.list_foundation_models()

        # Step 2: Filter to ON_DEMAND and INFERENCE_PROFILE models only
        filtered_models = self.filter_on_demand_models(all_models)

        # Step 3: Check which models need access
        accessible_models, models_needing_access = self.check_model_access_status(filtered_models)

        # Step 4 & 5: For models needing access, get offers and create agreements
        activation_results = {
            'total_models': len(all_models),
            'filtered_models': len(filtered_models),
            'already_accessible': len(accessible_models),
            'attempted_activation': len(models_needing_access),
            'successful_activations': 0,
            'failed_activations': 0,
            'activation_details': []
        }

        for model in models_needing_access:
            model_id = model.get('modelId')
            model_name = model.get('modelName', 'Unknown')

            logger.info(f"Processing model: {model_name} ({model_id})")

            # Get agreement offers for this model
            offers = self.get_agreement_offers(model_id)

            if not offers:
                logger.warning(f"No offers found for model {model_id}")
                activation_results['failed_activations'] += 1
                activation_results['activation_details'].append({
                    'model_id': model_id,
                    'model_name': model_name,
                    'status': 'failed',
                    'reason': 'no_offers_available'
                })
                continue

            # Try to create agreement with the first available offer
            offer = offers[0]
            offer_token = offer.get('offerToken')

            if not offer_token:
                logger.warning(f"No offer token found for model {model_id}")
                activation_results['failed_activations'] += 1
                activation_results['activation_details'].append({
                    'model_id': model_id,
                    'model_name': model_name,
                    'status': 'failed',
                    'reason': 'no_offer_token'
                })
                continue

            # Create the model agreement
            success = self.create_model_agreement(model_id, offer_token)

            if success:
                activation_results['successful_activations'] += 1
                activation_results['activation_details'].append({
                    'model_id': model_id,
                    'model_name': model_name,
                    'status': 'success',
                    'offer_token': offer_token
                })
            else:
                activation_results['failed_activations'] += 1
                activation_results['activation_details'].append({
                    'model_id': model_id,
                    'model_name': model_name,
                    'status': 'failed',
                    'reason': 'agreement_creation_failed'
                })

        return activation_results

    def print_summary(self, results: Dict[str, Any]):
        """
        Print a summary of the activation results

        Args:
            results: Results dictionary from activate_all_models
        """
        print("\n" + "="*50)
        print("BEDROCK MODEL ACTIVATION SUMMARY")
        print("="*50)
        print(f"Region: {self.region_name}")
        print(f"Total models found: {results['total_models']}")
        print(f"Filtered models (ON_DEMAND/INFERENCE_PROFILE): {results['filtered_models']}")
        print(f"Already accessible: {results['already_accessible']}")
        print(f"Attempted activation: {results['attempted_activation']}")
        print(f"Successful activations: {results['successful_activations']}")
        print(f"Failed activations: {results['failed_activations']}")

        if results['activation_details']:
            print("\nACTIVATION DETAILS:")
            print("-" * 30)
            for detail in results['activation_details']:
                status_symbol = "✓" if detail['status'] == 'success' else "✗"
                print(f"{status_symbol} {detail['model_name']} ({detail['model_id']})")
                if detail['status'] == 'failed':
                    print(f"   Reason: {detail['reason']}")

        print("="*50)
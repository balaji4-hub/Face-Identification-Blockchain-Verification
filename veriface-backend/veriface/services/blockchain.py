"""
STAGE 8: BLOCKCHAIN
Handles evidence hashing, timestamping, and transaction recording
"""
import time
import json
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from veriface.core.config import settings
from veriface.models.database import AsyncSession
from veriface.models.schemas import VerificationSession, BlockchainRecord, BlockchainBlock
from veriface.models.schemas_api import BlockchainResponse, BlockchainTransaction


@dataclass
class Block:
    """Blockchain block structure"""
    index: int
    timestamp: datetime
    evidence_hash: str
    previous_hash: str
    hash: str
    nonce: int = 0
    transactions: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.transactions is None:
            self.transactions = []
    
    def calculate_hash(self) -> str:
        """Calculate block hash"""
        block_data = f"{self.index}{self.timestamp.isoformat()}{self.evidence_hash}{self.previous_hash}{self.nonce}"
        return hashlib.sha256(block_data.encode()).hexdigest()


class BlockchainService:
    """Simplified blockchain for evidence timestamping"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.difficulty = settings.difficulty
        self.genesis_timestamp = settings.genesis_timestamp
    
    async def get_chain(self) -> List[Block]:
        """Get current blockchain"""
        result = await self.session.execute(
            select(BlockchainBlock).order_by(BlockchainBlock.index)
        )
        blocks = result.scalars().all()
        
        if not blocks:
            return []
        
        return [
            Block(
                index=b.index,
                timestamp=b.timestamp,
                evidence_hash=b.evidence_hash,
                previous_hash=b.previous_hash,
                hash=b.hash,
                nonce=b.nonce,
                transactions=[]
            )
            for b in blocks
        ]
    
    async def get_latest_block(self) -> Optional[Block]:
        """Get latest block in chain"""
        result = await self.session.execute(
            select(BlockchainBlock)
            .order_by(BlockchainBlock.index.desc())
            .limit(1)
        )
        block = result.scalar_one_or_none()
        
        if not block:
            return None
        
        return Block(
            index=block.index,
            timestamp=block.timestamp,
            evidence_hash=block.evidence_hash,
            previous_hash=block.previous_hash,
            hash=block.hash,
            nonce=block.nonce
        )
    
    def proof_of_work(self, block: Block) -> tuple:
        """
        Simple proof of work algorithm
        
        Returns:
            Tuple of (nonce, hash)
        """
        nonce = 0
        while True:
            block.nonce = nonce
            block_hash = block.calculate_hash()
            
            # Check if hash meets difficulty requirement
            if block_hash[:self.difficulty] == '0' * self.difficulty:
                return nonce, block_hash
            
            nonce += 1
    
    def create_genesis_block(self) -> Block:
        """Create genesis block"""
        genesis = Block(
            index=0,
            timestamp=datetime.fromtimestamp(self.genesis_timestamp),
            evidence_hash="0" * 64,
            previous_hash="0" * 64,
            hash="",
            nonce=0
        )
        nonce, block_hash = self.proof_of_work(genesis)
        genesis.hash = block_hash
        return genesis
    
    async def add_block(
        self, 
        evidence_hash: str,
        session_id: str,
        merkle_root: str
    ) -> BlockchainRecord:
        """
        Add new block to blockchain
        
        Args:
            evidence_hash: Hash of evidence to record
            session_id: Verification session ID
            merkle_root: Merkle root of evidence
            
        Returns:
            BlockchainRecord
        """
        latest_block = await self.get_latest_block()
        
        if not latest_block:
            # Create genesis block
            latest_block = self.create_genesis_block()
            
            # Save genesis block
            genesis_db = BlockchainBlock(
                index=latest_block.index,
                timestamp=latest_block.timestamp,
                evidence_hash=latest_block.evidence_hash,
                previous_hash=latest_block.previous_hash,
                hash=latest_block.hash,
                nonce=latest_block.nonce,
                transactions_count=1
            )
            self.session.add(genesis_db)
        
        # Create new block
        new_block = Block(
            index=latest_block.index + 1,
            timestamp=datetime.utcnow(),
            evidence_hash=evidence_hash,
            previous_hash=latest_block.hash,
            hash="",
            nonce=0
        )
        
        # Proof of work
        nonce, block_hash = self.proof_of_work(new_block)
        new_block.hash = block_hash
        
        # Save block
        block_db = BlockchainBlock(
            index=new_block.index,
            timestamp=new_block.timestamp,
            evidence_hash=new_block.evidence_hash,
            previous_hash=new_block.previous_hash,
            hash=new_block.hash,
            nonce=new_block.nonce,
            transactions_count=1
        )
        self.session.add(block_db)
        
        # Create transaction record
        transaction_hash = hashlib.sha256(
            f"{new_block.index}{new_block.evidence_hash}{time.time()}".encode()
        ).hexdigest()
        
        record = BlockchainRecord(
            id=str(uuid.uuid4()),
            session_id=session_id,
            block_number=new_block.index,
            block_hash=new_block.hash,
            transaction_hash=transaction_hash,
            evidence_hash=evidence_hash,
            previous_hash=latest_block.hash,
            merkle_root=merkle_root,
            timestamp=datetime.utcnow(),
            confirmed=True,
            confirmations=1,
            extra_data={
                "nonce": nonce,
                "difficulty": self.difficulty
            }
        )
        self.session.add(record)
        
        return record
    
    async def record_evidence(
        self,
        session_id: str,
        evidence_hash: str,
        merkle_root: str
    ) -> BlockchainResponse:
        """
        Record evidence on blockchain
        
        Args:
            session_id: Verification session ID
            evidence_hash: Hash of evidence
            merkle_root: Merkle root of evidence
            
        Returns:
            BlockchainResponse
        """
        start_time = time.time()
        
        # Add block to chain
        record = await self.add_block(evidence_hash, session_id, merkle_root)
        
        # Update session
        session = (await self.session.execute(
            select(VerificationSession).where(VerificationSession.id == session_id)
        )).scalar_one()
        session.status = "blockchain_recorded"
        
        await self.session.commit()
        
        processing_time = (time.time() - start_time) * 1000
        
        transaction = BlockchainTransaction(
            transaction_hash=record.transaction_hash,
            block_number=record.block_number,
            block_hash=record.block_hash,
            timestamp=record.timestamp,
            evidence_hash=record.evidence_hash,
            previous_hash=record.previous_hash,
            confirmations=record.confirmations
        )
        
        return BlockchainResponse(
            session_id=session_id,
            transaction=transaction,
            merkle_root=merkle_root,
            recorded=True,
            recording_time_ms=processing_time
        )
    
    async def verify_chain(self) -> Dict[str, Any]:
        """Verify blockchain integrity"""
        chain = await self.get_chain()
        
        if len(chain) == 0:
            return {"valid": False, "reason": "Chain is empty"}
        
        # Verify genesis
        if chain[0].index != 0:
            return {"valid": False, "reason": "Invalid genesis block"}
        
        # Verify each block
        for i in range(1, len(chain)):
            current = chain[i]
            previous = chain[i - 1]
            
            # Verify hash
            if current.hash != current.calculate_hash():
                return {"valid": False, "reason": f"Invalid hash at block {i}"}
            
            # Verify previous hash reference
            if current.previous_hash != previous.hash:
                return {"valid": False, "reason": f"Broken chain at block {i}"}
        
        # Verify proof of work
        for block in chain:
            if block.hash[:self.difficulty] != '0' * self.difficulty:
                return {"valid": False, "reason": f"Invalid proof of work at block {block.index}"}
        
        return {
            "valid": True,
            "block_count": len(chain),
            "latest_hash": chain[-1].hash if chain else None
        }
    
    async def get_transaction(self, session_id: str) -> Optional[BlockchainRecord]:
        """Get blockchain record for a session"""
        result = await self.session.execute(
            select(BlockchainRecord).where(BlockchainRecord.session_id == session_id)
        )
        return result.scalar_one_or_none()


# Helper imports
import uuid
from sqlalchemy import select
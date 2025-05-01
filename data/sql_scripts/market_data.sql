create database market_data;
use market_data;
drop table test_bnc_kline_1m;
create table bnc_kline_1m (
    ts0 bigint,
    symbol varchar(20),
    ts1 bigint,
    t datetime,
    o double,
    h double,
    l double,
    c double,
    v double,
    q double,
    bv double,
    bq double,
    n int,
    primary key(ts0, symbol)
);
drop table bnc_kline_1m;
create table bnc_kline_5m (
    ts0 bigint,
    symbol varchar(20),
    ts1 bigint,
    t datetime,
    o double,
    h double,
    l double,
    c double,
    v double,
    q double,
    bv double,
    bq double,
    n int,
    primary key(ts0, symbol)
);
create table bnc_kline_5m (
    ts0 bigint,
    symbol varchar(20),
    ts1 bigint,
    t datetime,
    o double,
    h double,
    l double,
    c double,
    v double,
    q double,
    bv double,
    bq double,
    n int,
    primary key(ts0, symbol)
);
create table test_bnc_kline_15m (
    ts0 bigint,
    symbol varchar(20),
    ts1 bigint,
    ts_event bigint,
    ts_insert bigint,
    t datetime,
    o double,
    h double,
    l double,
    c double,
    v double,
    q double,
    bv double,
    bq double,
    n int
);
create table test_bnc_kline_5m (
    ts0 bigint,
    symbol varchar(20),
    ts1 bigint,
    ts_event bigint,
    ts_insert bigint,
    t datetime,
    o double,
    h double,
    l double,
    c double,
    v double,
    q double,
    bv double,
    bq double,
    n int
);
create table test_bnc_kline_1m (
    ts0 bigint,
    symbol varchar(20),
    ts1 bigint,
    ts_event bigint,
    ts_insert bigint,
    t datetime,
    o double,
    h double,
    l double,
    c double,
    v double,
    q double,
    bv double,
    bq double,
    n int,
    primary key (ts0, symbol, ts_insert)
);
drop table bnc_aggtrades;
create table bnc_aggtrades (
	ts bigint,
    symbol varchar(20),
    a bigint,
    t datetime,
    f bigint, 
    l bigint, 
    p double,
    q double,
    v double,
    n int,
    m bool,
    primary key (a, symbol),
    index idx_ts (ts),
    index idx_symbol (symbol)
);
create table bnc_aggtrades_record (
    symbol varchar(20),
    a0 bigint,
    a1 bigint,
    ts0 bigint,
	ts1 bigint,
    primary key (symbol)
);
select 
	* 
from 
	bnc_global_ls_acct_ratio_5m;
create table test_bnc_aggtrades (
    ts bigint,
    symbol varchar(20),
    t datetime,
    ets bigint,
    its bigint, -- insert timestamp
    atid bigint,
    p double,
    q double,
    v double,
    tid0 bigint,
    tid1 bigint,
    m bool,
    primary key (ts, symbol)
);
drop table test_bnc_kline_15m;
drop table test_bnc_aggtrades;
select STD(bv / q - bq / v) from bnc_kline_15m; 
alter table bnc_kline_15m rename column bv to a;
alter table bnc_kline_15m rename column bq to bv;
alter table bnc_kline_15m rename column a to bq;
alter table bnc_kline_1m rename column bv to a;
alter table bnc_kline_1m rename column bq to bv;
alter table bnc_kline_1m rename column a to bq;
create table bnc_top_ls_acct_ratio_5m (
	ts0 bigint,
    symbol varchar(20),
    l double,
    s double,
    lsr double,
    primary key (ts0, symbol)
);
create table bnc_top_ls_pos_ratio_5m (
	ts0 bigint,
    symbol varchar(20),
    l double,
    s double,
    lsr double,
    primary key (ts0, symbol)
);
create table bnc_global_ls_acct_ratio_5m (
	ts0 bigint,
    symbol varchar(20),
    l double,
    s double,
    lsr double,
    primary key (ts0, symbol)
);
create table bnc_oi_5m (
	ts0 bigint,
    symbol varchar(20),
    oi double,
    oiv double,
    primary key (ts0, symbol)
);
create table bnc_taker_ls_ratio_5m (
	ts0 bigint,
    symbol varchar(20),
    l double,
    s double,
    lsr double,
    primary key (ts0, symbol)
);
create table test_bnc_oi_5m (
	ts0 bigint,
    symbol varchar(20),
    its bigint,
    oi double,
    oiv double,
    primary key (ts0, symbol)
);
create table test_bnc_taker_ls_ratio_5m (
	ts0 bigint,
    symbol varchar(20),
    its bigint,
    l double,
    s double,
    lsr double,
    primary key (ts0, symbol)
);
create table test_bnc_top_ls_acct_ratio_5m (
	ts0 bigint,
    symbol varchar(20),
    its bigint,
    l double,
    s double,
    lsr double,
    primary key (ts0, symbol)
);
create table test_bnc_top_ls_pos_ratio_5m (
	ts0 bigint,
    symbol varchar(20),
    its bigint,
    l double,
    s double,
    lsr double,
    primary key (ts0, symbol)
);
create table test_bnc_global_ls_acct_ratio_5m (
	ts0 bigint,
    symbol varchar(20),
    its bigint,
    l double,
    s double,
    lsr double,
    primary key (ts0, symbol)
);
create table bnc_depth5_30s (
	ts bigint,
	symbol varchar(20),
	`level` int,
	p double,
	v double,
	q double,
	primary key (ts, symbol, `level`)
);
create table bnc_depth5_30s_record (
	symbol varchar(20),	
	`date` date,
	primary key(symbol, `date`)
);
create table bnc_aggtrades_from_website_record (
	`date` date,
	symbol varchar(20),
	ts bigint,
	primary key (`date`, symbol)
);
create table bnc_bookdepth5_from_website_record (
	`date` date,
	symbol varchar(20),
	ts bigint,
	primary key (`date`, symbol)
);
CREATE TABLE bnc_symbols (
    symbol VARCHAR(20) PRIMARY KEY,
    pair VARCHAR(20),
    geckoName VARCHAR(40),
    contractType VARCHAR(20),
    deliveryDate BIGINT,
    onboardDate BIGINT,
    status VARCHAR(20),
    maintMarginPercent VARCHAR(20),
    requiredMarginPercent VARCHAR(20),
    baseAsset VARCHAR(20),
    quoteAsset VARCHAR(20),
    marginAsset VARCHAR(20),
    pricePrecision INT,
    quantityPrecision INT,
    baseAssetPrecision INT,
    quotePrecision INT,
    underlyingType VARCHAR(20),
    underlyingSubType VARCHAR(100),
    triggerProtect VARCHAR(20),
    liquidationFee VARCHAR(20),
    marketTakeBound VARCHAR(20),
    maxMoveOrderLimit INT
);
create table gecko_market_cap_1d (
	ts bigint,
	symbol varchar(20),
	t date,
	price double,
	market_cap double,
	vol double,
	primary key (ts, symbol)
);
delete from gecko_market_cap_1d;
drop table bnc_symbols;
drop table bnc_aggtrades_record;
drop table test_bnc_oi_5m;
drop table test_bnc_taker_ls_ratio_5m;
select 
    ts0, symbol, oi, oiv 
from 
    test_bnc_oi_5m;
replace into bnc_global_ls_acct_ratio_5m  
    (ts0, symbol, l, s, lsr)
select 
    ts0, symbol, l, s, lsr 
from 
    test_bnc_global_ls_acct_ratio_5m;
replace into bnc_top_ls_acct_ratio_5m 
    (ts0, symbol, l, s, lsr)
select 
    ts0, symbol, l, s, lsr 
from 
    test_bnc_top_ls_acct_ratio_5;
replace into bnc_top_ls_pos_ratio_5m  
    (ts0, symbol, l, s, lsr)
select 
    ts0, symbol, oi, oiv 
from 
    test_bnc_top_ls_pos_ratio_5m;
alter table bnc_top_ls_acct_ratio_5m add primary key (ts0, symbol);
select 
    ts_insert, (ts_insert - ts_event)
from 
    test_bnc_kline_5m tbkm
order by
    ts_insert desc ;
select 
    ts_insert, 
    ts_insert - ts_event,
    avg(ts_insert - ts_event) over (order by ts_insert ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT row)
from 
    test_bnc_kline_5m tbkm
order by
	ts_insert desc;
delete from test_bnc_kline_5m;
select 
    * 
from 
    bnc_kline_15m bkm 
where 
	ts0 > unix_timestamp("2024-09-08 12:00:00")* 1000;
-- 读取bnc_kline_15m最新的200条数据	
select 
    * 
from 
    bnc_kline_15m bkm 
order by
    ts0 desc
limit 200;
-- 读取bnc_aggtrades最新的200条数据
select
    * 
from 
    bnc_aggtrades
order by
    ts desc
limit
    200;
select 
   avg(delta_ts)
from
	select
	    its - ets delta_ts
	from 
	    test_bnc_aggtrades
	order by
	    ts desc
	limit
	    200;
select 
    * 
from 
    bnc_global_ls_acct_ratio_5m
order by
    ts0 desc 
limit 
    10;
select 
    from_unixtime(1720782600000); 
-- 测试aggtrades的下载效率
select 
    ets, 
    avg(ets - ts) over (order by ets ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT row)
from 
    test_bnc_aggtrades tba
order by
    ets desc 
limit 
    20;
select 
    max(ets - ts)
from 
    test_bnc_aggtrades tba;
select 
	*
from 
	bnc_kline_15m bkm
where 
	symbol = 'ETHUSDT'
order by
	ts0 desc 
limit 100;
select
	sum(p * q)
from (
	select 
		*
	from 
		bnc_depth5_30s bds 
	limit 100
) s;
select 
	*, bq / q, bv / v
from 
	bnc_kline_15m bkm 
where 
	bq / q > 1 or bv / v > 1;
select 
	avg(ts_delta)
from (
	select 
		ts0, symbol, min(ts_insert) - ts0 ts_delta
	from
		test_bnc_kline_1m tbkm
	where 
		ts0 > 1731863400000
	group by
		ts0, symbol
) s;
select
	*
from
	bnc_kline_5m bkm
order by
	ts0 desc
limit 
	200;
select 
	max(bq / q), min(bv / v)
from 
	(select bq, q, bv, v from bnc_kline_5m limit 1000000) s;
select
	from_unixtime(max(ts) / 1000)
from
	bnc_depth5_30s bds;
select 
    *, a1 - a0 num
from
	bnc_aggtrades_record bar
-- where
--     substr(symbol, -4, -1) == "USDT" 
-- 	symbol IN ("TLMUSDT", "ETHUSDT", "SOLUSDT")
order by
	num asc;
select *, substr(symbol, -4, 4) from bnc_aggtrades_record;
select @@innodb_buffer_pool_size;
delete from bnc_aggtrades_from_website_record ;
select * from bnc_aggtrades_from_website_record where symbol = 'BTCUSDT' and `date` = '2025-01-09';